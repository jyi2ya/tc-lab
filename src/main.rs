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

#[allow(dead_code)]
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
        let _timer = ScopeTimer::with_label("read and build");

        let args: Vec<_> = std::env::args().collect();
        let filename = &args[1];
        let file = File::open(filename).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let content = &mmap[..];
        let content = unsafe { std::str::from_utf8_unchecked(content) };

        let edges: Vec<_> = {
            let _timer = ScopeTimer::with_label("scan and parse");
            content
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
                    Some((src.max(dst), src.min(dst)))
                })
                .collect()
        };

        let edges = {
            let _timer = ScopeTimer::with_label("sort and dedup");
            let mut edges = edges;
            edges.as_mut_slice().par_sort_unstable();
            edges.dedup();
            edges
        };

        edges
    };

    let max_node_id = edges_group_by_src.last().unwrap().0 as usize;

    let lowers = {
        let _timer = ScopeTimer::with_label("counting neighbors");
        let mut lowers: Vec<Vec<NodeId>> = Vec::new();
        lowers.resize(max_node_id + 1, Vec::default());

        for (src, dst) in edges_group_by_src {
            lowers[src as usize].push(dst);
        }

        lowers
    };

    const TRUNK_SIZE_RATIO: usize = 32;

    let result: usize = {
        let _timer = ScopeTimer::with_label("computation");
        lowers
            .as_slice()
            .par_chunks(lowers.len() / (TRUNK_SIZE_RATIO * rayon::current_num_threads()))
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
                        .flat_map(|second| lowers[*second as usize].iter())
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
