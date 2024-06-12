#![feature(slice_split_once)]

use bitvec::prelude::*;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::fs::File;
use std::time;
use voracious_radix_sort::RadixSort;

use bitvec::bitvec;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
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
type FatNodeId = u64;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct Edge(NodeId, NodeId);

impl From<Edge> for FatNodeId {
    fn from(value: Edge) -> Self {
        ((value.0 as FatNodeId) << 32) | (value.1 as FatNodeId)
    }
}

impl From<FatNodeId> for Edge {
    fn from(value: FatNodeId) -> Self {
        Self((value >> 32) as NodeId, value as NodeId)
    }
}

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
                .map(str::as_bytes)
                .filter_map(|line| line.split_once(|x| x.is_ascii_whitespace()))
                .filter_map(|(src, dst)| {
                    let src: NodeId = atoi_simd::parse(src).ok()?;
                    let dst: NodeId = atoi_simd::parse(dst).ok()?;
                    match NodeId::cmp(&src, &dst) {
                        Ordering::Greater => Some(FatNodeId::from(Edge(src, dst))),
                        Ordering::Equal => None,
                        Ordering::Less => Some(FatNodeId::from(Edge(dst, src))),
                    }
                })
                .collect()
        };

        let edges = {
            let _timer = ScopeTimer::with_label("sort and dedup");
            let mut edges = edges;
            edges.voracious_mt_sort(rayon::current_num_threads());
            edges.dedup();
            edges
        };

        edges
    };

    let max_node_id = edges_group_by_src.last().map(|x| x >> 32).unwrap() as usize;

    let lowers = {
        let _timer = ScopeTimer::with_label("counting neighbors");
        let mut lowers: Vec<SmallVec<[NodeId; 16]>> = Vec::new();
        lowers.resize(max_node_id + 1, SmallVec::default());

        for Edge(src, dst) in edges_group_by_src.into_iter().map(Edge::from) {
            lowers[src as usize].push(dst);
        }

        lowers.into_boxed_slice()
    };

    const TRUNK_SIZE_RATIO: usize = 32;

    let result: usize = {
        let _timer = ScopeTimer::with_label("computation");
        lowers
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
