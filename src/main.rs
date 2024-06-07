#![feature(slice_split_once)]

use bitvec::prelude::*;
use std::fs::File;
use std::time;

use bitvec::bitvec;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::slice::ParallelSliceMut;
use rayon::str::ParallelString;

struct ScopeTimer {
    label: String,
    start: time::Instant,
}

impl ScopeTimer {
    pub fn with_label(label: impl ToString) -> Self {
        Self {
            label: label.to_string(),
            start: time::Instant::now(),
        }
    }

    pub fn stop(self) {}
}

impl Drop for ScopeTimer {
    fn drop(&mut self) {
        let end = time::Instant::now();
        println!(
            "[ScopeTimer] {} seconds | {}",
            (end - self.start).as_secs_f64(),
            self.label,
        );
    }
}

type NodeId = u32;

fn main() {
    let _timer = ScopeTimer::with_label("totals");

    let edges_group_by_src = {
        let _timer = ScopeTimer::with_label("read and parse");

        let args: Vec<_> = std::env::args().collect();
        let filename = &args[1];
        let file = File::open(filename).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let content = &mmap[..];
        let content = unsafe { std::str::from_utf8_unchecked(content) };

        let mut edges: Vec<_> = content
            .par_lines()
            .filter_map(|line| {
                let line = line.as_bytes();
                if line[0] == b'#' {
                    return None;
                }
                let (src, dst) = line.split_once(|x| x.is_ascii_whitespace()).unwrap();
                let src: NodeId = atoi_simd::parse(src).unwrap();
                let dst: NodeId = atoi_simd::parse(dst).unwrap();
                if src == dst {
                    return None;
                }
                Some((src.min(dst), src.max(dst)))
            })
            .collect();
        edges.as_mut_slice().par_sort_unstable();
        edges.dedup();
        edges
    };

    // let edges_group_by_dst_handle = {
    //     let mut edges = edges_group_by_src.clone();
    //     thread::spawn(move || {
    //         let _timer = ScopeTimer::with_label("sort by dst");
    //         edges
    //             .as_mut_slice()
    //             .par_sort_unstable_by_key(|(src, dst)| (*dst, *src));
    //         edges
    //     })
    // };

    let timer_pre_computing = ScopeTimer::with_label("counting neighbors");

    let max_node_id = edges_group_by_src
        .iter()
        .map(|(_, bigger)| bigger)
        .max()
        .cloned()
        .map(|x| x as usize)
        .unwrap();

    let uppers = {
        let mut uppers: Vec<Vec<NodeId>> = Vec::new();
        uppers.resize(max_node_id + 1, Vec::default());

        for (src, dst) in edges_group_by_src {
            uppers[src as usize].push(dst);
        }

        uppers
    };

    timer_pre_computing.stop();

    // let lowers_handle = thread::spawn(move || {
    //     let mut lowers: Vec<Vec<NodeId>> = Vec::new();
    //     lowers.resize(max_node_id + 1, Vec::default());

    //     for (src, dst) in edges_group_by_dst_handle.join().unwrap() {
    //         lowers[dst as usize].push(src);
    //     }

    //     lowers
    // });

    // let uppers = uppers_handle.join().unwrap();
    // let _lowers = lowers_handle.join().unwrap();

    const TRUNK_SIZE_RATIO: usize = 32;

    let result: usize = {
        let _timer = ScopeTimer::with_label("computation");
        uppers
            .as_slice()
            .par_chunks(uppers.len() / (TRUNK_SIZE_RATIO * rayon::current_num_threads()))
            .map(|data| {
                let mut bitmap = bitvec![];
                bitmap.resize(max_node_id + 1, false);
                let mut bitmap = bitmap.into_boxed_bitslice();

                let mut result = 0_usize;
                for first in data {
                    for neigh in first {
                        bitmap.set(*neigh as usize, true);
                    }

                    result += first
                        .iter()
                        .flat_map(|second| uppers[*second as usize].iter())
                        .map(|third| bitmap[*third as usize] as usize)
                        .sum::<usize>();

                    for neigh in first {
                        bitmap.set(*neigh as usize, false);
                    }
                }

                result
            })
            .sum()
    };

    println!("{result}");
}
