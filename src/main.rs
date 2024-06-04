use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::thread;
use std::time;

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

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
type EdgeSet = HashSet<(NodeId, NodeId)>;
type Graph = Vec<HashSet<NodeId>>;

fn load_graph() -> (EdgeSet, Graph) {
    let args: Vec<_> = std::env::args().collect();
    let filename = &args[1];
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let (reader_tx, reader_rx) = std::sync::mpsc::channel();
    let (edge_tx, edge_rx) = std::sync::mpsc::channel();
    let (graph_tx, graph_rx) = std::sync::mpsc::channel();

    let reader_thread = thread::spawn(move || {
        let _timer = ScopeTimer::with_label("reader thread");
        for line in reader.lines() {
            let line = line.unwrap();
            if line.starts_with('#') {
                continue;
            }

            let mut splited = line.split_whitespace();
            let src: usize = splited.next().unwrap().parse().unwrap();
            let dst: usize = splited.next().unwrap().parse().unwrap();
            if src == dst {
                continue;
            }

            reader_tx.send((src, dst)).unwrap();
        }
    });

    let bridge_thread = thread::spawn(move || {
        let _timer = ScopeTimer::with_label("bridge thread");
        let mut compact: HashMap<usize, NodeId> = HashMap::new();
        while let Ok((src, dst)) = reader_rx.recv() {
            let next_index = compact.len() as NodeId;
            let src = *compact.entry(src).or_insert(next_index);
            let next_index = compact.len() as NodeId;
            let dst = *compact.entry(dst).or_insert(next_index);

            edge_tx.send((src, dst)).unwrap();
            graph_tx.send((src, dst)).unwrap();
        }
    });

    let edge_thread = thread::spawn(move || {
        let _timer = ScopeTimer::with_label("edge thread");
        let mut edges: HashSet<(NodeId, NodeId)> = HashSet::new();
        while let Ok((src, dst)) = edge_rx.recv() {
            let (src, dst) = (src.min(dst), src.max(dst));
            edges.insert((src, dst));
        }
        edges
    });

    let graph_thread = thread::spawn(move || {
        let _timer = ScopeTimer::with_label("graph thread");
        let mut graph: Vec<HashSet<NodeId>> = Vec::new();
        while let Ok((src, dst)) = graph_rx.recv() {
            let size = (src.max(dst) + 1) as usize;
            (graph.len() < size).then(|| graph.resize(size, HashSet::new()));
            graph[src as usize].insert(dst);
            graph[dst as usize].insert(src);
        }
        graph
    });

    reader_thread.join().unwrap();
    bridge_thread.join().unwrap();
    let edges = edge_thread.join().unwrap();
    let graph = graph_thread.join().unwrap();

    (edges, graph)
}

fn main() {
    let _timer = ScopeTimer::with_label("totals");

    let (edges, graph) = {
        let _timer = ScopeTimer::with_label("loading data");
        let (edges, graph) = load_graph();
        (edges, graph)
    };

    let accumulate = {
        let _timer = ScopeTimer::with_label("computation");
        let result: usize = edges
            .par_iter()
            .map(|(src, dst)| {
                graph[*src as usize]
                    .intersection(&graph[*dst as usize])
                    .count()
            })
            .sum();
        result
    };

    let result = accumulate / 3;
    println!("{result}");
}
