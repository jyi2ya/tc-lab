#![feature(stdarch_x86_avx512)]
#![feature(slice_split_once)]
#![feature(slice_partition_dedup)]

use std::arch::x86_64::*;
use std::cmp::Ordering;
use std::fs::File;
use std::sync::Mutex;
use std::time;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

fn simd_parse_u32(input: &[u8]) -> u32 {
    unsafe {
        let len = input.len();
        let mem = input.as_ptr() as *const i8;
        let trunc_mask = ((1_u64 << len) - 1) as u16;
        let data = _mm_maskz_loadu_epi8(trunc_mask, mem);
        let sub = _mm_set1_epi8(b'0' as i8);
        let data = _mm_maskz_sub_epi8(trunc_mask, data, sub);
        let data = match len {
            0 => _mm_bslli_si128::<16>(data),
            1 => _mm_bslli_si128::<15>(data),
            2 => _mm_bslli_si128::<14>(data),
            3 => _mm_bslli_si128::<13>(data),
            4 => _mm_bslli_si128::<12>(data),
            5 => _mm_bslli_si128::<11>(data),
            6 => _mm_bslli_si128::<10>(data),
            7 => _mm_bslli_si128::<9>(data),
            8 => _mm_bslli_si128::<8>(data),
            9 => _mm_bslli_si128::<7>(data),
            10 => _mm_bslli_si128::<6>(data),
            11 => _mm_bslli_si128::<5>(data),
            12 => _mm_bslli_si128::<4>(data),
            13 => _mm_bslli_si128::<3>(data),
            14 => _mm_bslli_si128::<2>(data),
            15 => _mm_bslli_si128::<1>(data),
            16 => _mm_bslli_si128::<0>(data),
            _ => unreachable!(),
        };
        let data = _mm512_cvtepi8_epi32(data);
        let weigh = _mm512_set_epi32(
            1,
            10,
            100,
            1_000,
            10_000,
            100_000,
            1000_000,
            10_000_000,
            100_000_000,
            1000_000_000,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let result = _mm512_mullo_epi32(data, weigh);
        _mm512_reduce_add_epi32(result) as u32
    }
}

struct Avx512Lines<'a> {
    res: &'a [u8],
}

impl<'a> From<&'a [u8]> for Avx512Lines<'a> {
    fn from(res: &'a [u8]) -> Self {
        Self { res }
    }
}

impl<'a> Iterator for Avx512Lines<'a> {
    type Item = Option<(&'a [u8], &'a [u8])>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.res.is_empty() {
            return None;
        }
        unsafe {
            let mem = self.res.as_ptr() as *const i32;
            let data = if self.res.len() >= 64 {
                _mm512_loadu_epi32(mem)
            } else {
                let trunc_mask = ((1_usize << self.res.len()) - 1) as u16;
                _mm512_maskz_loadu_epi32(trunc_mask, mem)
            };
            let endl_pos = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(b'\n' as i8))
                .trailing_zeros() as usize;
            if endl_pos == 64 {
                self.res = &self.res[std::cmp::min(endl_pos, self.res.len())..];
                Some(None)
            } else {
                let line = &self.res[..endl_pos];
                let line = line.strip_suffix(&[b'\r']).unwrap_or(line);
                let tab_pos = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8(b'\t' as i8))
                    .trailing_zeros() as usize;
                let result = if tab_pos + 1 <= line.len() {
                    let (src, dst) = (&line[..tab_pos], &line[tab_pos + 1..]);
                    if src[0].is_ascii_digit() && dst[0].is_ascii_digit() {
                        Some((src, dst))
                    } else {
                        None
                    }
                } else {
                    None
                };

                self.res = &self.res[std::cmp::min(endl_pos + 1, self.res.len())..];
                Some(result)
            }
        }
    }
}

struct Avx512Bitmap {
    data: Vec<u32>,
}

impl std::fmt::Debug for Avx512Bitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in &self.data {
            writeln!(f, "{:032b} ", i)?;
        }
        writeln!(f, "\n")
    }
}

impl Avx512Bitmap {
    #[inline]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![0; capacity],
        }
    }

    #[inline]
    fn batch_flip(&mut self, idx: &[u32]) {
        idx.iter()
            .map(|dest| (dest >> 5, dest & 0x1F))
            .for_each(|(idx, offset)| self.data[idx as usize] ^= 1_u32 << offset);
    }

    #[inline]
    fn batch_count(&self, idx: &[u32]) -> u32 {
        idx.chunks(16)
            .map(|chunk| unsafe {
                let mem = chunk.as_ptr() as *const i32;
                let bits = match chunk.len() {
                    16 => {
                        let target = _mm512_loadu_epi32(mem);

                        let idx = _mm512_srli_epi32(target, 5);
                        let offset = _mm512_and_si512(target, _mm512_set1_epi32(0x1F));

                        let buf = _mm512_i32gather_epi32::<4>(idx, self.data.as_ptr() as *const u8);
                        let mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), offset);
                        _mm512_and_si512(buf, mask)
                    }
                    _ => {
                        let trunc_mask = ((1 << chunk.len()) - 1) as u16;
                        let target = _mm512_maskz_loadu_epi32(trunc_mask, mem);

                        let idx = _mm512_srli_epi32(target, 5);
                        let offset = _mm512_and_si512(target, _mm512_set1_epi32(0x1F));

                        let buf = _mm512_mask_i32gather_epi32::<4>(
                            _mm512_set1_epi32(0),
                            trunc_mask,
                            idx,
                            self.data.as_ptr() as *const u8,
                        );
                        let mask = _mm512_sllv_epi32(_mm512_set1_epi32(1), offset);
                        _mm512_and_si512(buf, mask)
                    }
                };
                let count = _mm512_cmpneq_epi32_mask(bits, _mm512_set1_epi32(0));
                count.count_ones()
            })
            .sum::<u32>()
    }
}

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
            "[ScopeTimer] {:<10.8} seconds | {}",
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
    #[inline(always)]
    fn from(value: Edge) -> Self {
        ((value.0 as FatNodeId) << 32) | (value.1 as FatNodeId)
    }
}

impl From<FatNodeId> for Edge {
    #[inline(always)]
    fn from(value: FatNodeId) -> Self {
        Self((value >> 32) as NodeId, value as NodeId)
    }
}

impl From<&FatNodeId> for Edge {
    #[inline(always)]
    fn from(value: &FatNodeId) -> Self {
        Self::from(*value)
    }
}

fn parallel_build_adjacency_list<'a>(
    lowers_offset: usize,
    lowers: &mut [&'a [NodeId]],
    mut src_list: &'a [NodeId],
    mut dst_list: &'a [NodeId],
) {
    const CUTOFF: usize = 5000 * 16;
    if src_list.len() < CUTOFF {
        while !src_list.is_empty() {
            let src = src_list[0];
            let split_pos = src_list
                .iter()
                .position(|&x| x != src)
                .unwrap_or(src_list.len());
            let (dst_current, dst_res) = dst_list.split_at(split_pos);
            let (_, src_res) = src_list.split_at(split_pos);
            lowers[src as usize - lowers_offset] = dst_current;
            (src_list, dst_list) = (src_res, dst_res);
        }
    } else {
        let mid = src_list.len() / 2;
        let mid_val = src_list[mid];
        let split_pos = src_list.partition_point(|&x| x < mid_val);
        let lowers_split_pos = mid_val as usize;
        let (lowers_l, lowers_r) = lowers.split_at_mut(lowers_split_pos - lowers_offset);
        let (src_l, src_r) = src_list.split_at(split_pos);
        let (dst_l, dst_r) = dst_list.split_at(split_pos);
        rayon::join(
            || parallel_build_adjacency_list(lowers_offset, lowers_l, src_l, dst_l),
            || parallel_build_adjacency_list(lowers_split_pos, lowers_r, src_r, dst_r),
        );
    }
}

fn main() {
    let compute_chunk_size_ratio = std::env::var("COMPUTE_CHUNK_SIZE_RATIO")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(8);
    println!("compute chunk size ratio: {compute_chunk_size_ratio}");

    let scan_chunk_size_ratio = std::env::var("SCAN_CHUNK_SIZE_RATIO")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(2);
    println!("scan chunk size ratio: {scan_chunk_size_ratio}");

    let _timer = ScopeTimer::with_label("totals");

    let edges_group_by_src = {
        let args: Vec<_> = std::env::args().collect();
        let filename = &args[1];
        let file = File::open(filename).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let content = &mmap[..];

        let edges: Vec<_> = {
            let _timer = ScopeTimer::with_label("scanning and parsing");
            let n_threads = scan_chunk_size_ratio * rayon::current_num_threads();
            let partition_size = content.len() / n_threads;
            let result = Mutex::new(Vec::new());
            rayon::scope(|s| {
                let mut content = content;
                while !content.is_empty() {
                    let cap = if content.len() < partition_size {
                        content.len()
                    } else {
                        content[partition_size..]
                            .iter()
                            .position(|&x| x == b'\n')
                            .map(|idx| idx + partition_size)
                            .unwrap_or(content.len())
                    };
                    let (current, res) = content.split_at(cap);
                    content = res;

                    s.spawn(|_| {
                        let mut local = Vec::with_capacity(partition_size / 20);
                        let extend = Avx512Lines::from(&current[..])
                            .flatten()
                            .map(|(src, dst)| (simd_parse_u32(src), simd_parse_u32(dst)))
                            .filter_map(|(src, dst)| match NodeId::cmp(&src, &dst) {
                                Ordering::Greater => Some(FatNodeId::from(Edge(src, dst))),
                                Ordering::Equal => None,
                                Ordering::Less => Some(FatNodeId::from(Edge(dst, src))),
                            });
                        local.extend(extend);
                        local.as_mut_slice().par_sort_unstable();
                        result.lock().unwrap().push(local);
                    });
                }
            });
            result.into_inner().unwrap()
        };

        let edges = {
            let _timer = ScopeTimer::with_label("merging thread results");
            let mut edges = edges;
            let sizes = edges.iter().map(Vec::len).collect::<Vec<_>>();
            let total_len = sizes.iter().sum::<usize>();
            let edges = edges.iter_mut().map(|x| x.as_slice()).collect::<Vec<_>>();

            let mut result: Vec<FatNodeId> = Vec::with_capacity(total_len);
            let spare = result.spare_capacity_mut();
            let spare: &mut [FatNodeId] = unsafe { std::mem::transmute(spare) };
            rayon_k_way_merge::merge(spare, edges);
            unsafe { result.set_len(total_len) };
            result
        };

        {
            let _timer = ScopeTimer::with_label("deduping edge list");

            let mut edges = edges;
            let chunk_size = edges.len() / rayon::current_num_threads() + 1;
            let mut parts = edges
                .par_chunks_mut(chunk_size)
                .map(|chunk| chunk.partition_dedup().0)
                .map(|x| {
                    #[allow(clippy::redundant_slicing)]
                    &x[..]
                })
                .collect::<Vec<_>>();

            let mut previous = parts[0].last().unwrap();
            for part in &mut parts[1..] {
                if let Some(pos) = part.iter().position(|x| x != previous) {
                    let (_, current) = part.split_at(pos);
                    *part = current;
                    previous = current.last().unwrap();
                } else {
                    *part = &mut [];
                }
            }
            let total_len = parts.iter().map(|x| x.len()).sum::<usize>();

            let mut result: Vec<FatNodeId> = Vec::with_capacity(total_len);
            let spare = result.spare_capacity_mut();
            let spare: &mut [FatNodeId] = unsafe { std::mem::transmute(spare) };
            rayon_k_way_merge::merge(spare, parts);
            unsafe { result.set_len(total_len) };
            result
        }
    };

    let max_node_id = edges_group_by_src.last().map(|x| x >> 32).unwrap() as usize;

    let (src_list, dst_list): (Vec<_>, Vec<_>) = {
        let _tiemr = ScopeTimer::with_label("adjusting memory layout");
        edges_group_by_src
            .into_par_iter()
            .map(|x| {
                let Edge(src, dst) = Edge::from(x);
                (src, dst)
            })
            .unzip()
    };

    let lowers = {
        let _timer = ScopeTimer::with_label("building adjacency list");
        let mut lowers: Vec<&[NodeId]> = Vec::new();
        lowers.resize_with(max_node_id + 1, Default::default);
        parallel_build_adjacency_list(0, lowers.as_mut_slice(), &src_list[..], &dst_list[..]);
        lowers.into_boxed_slice()
    };

    let result: usize = {
        let _timer = ScopeTimer::with_label("computation");
        lowers
            .par_chunks(lowers.len() / (compute_chunk_size_ratio * rayon::current_num_threads()))
            .map(|data| {
                let mut bitmap = Avx512Bitmap::with_capacity(max_node_id + 1);

                data.iter()
                    .filter_map(|&first| {
                        let min = first.first()?;
                        let max = first.last()?;
                        bitmap.batch_flip(first);

                        let result = first
                            .iter()
                            .map(|&idx| lowers[idx as usize])
                            .filter(|&second| match (second.first(), second.last()) {
                                (Some(first), Some(last)) => first <= max && last >= min,
                                _ => false,
                            })
                            .map(|idx| bitmap.batch_count(idx) as usize)
                            .sum::<usize>();

                        bitmap.batch_flip(first);

                        Some(result)
                    })
                    .sum::<usize>()
            })
            .sum()
    };

    println!("{result}");
}
