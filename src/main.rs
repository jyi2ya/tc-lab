use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::thread;
use std::time;

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

type EdgeSet = HashSet<(usize, usize)>;
type Graph = HashMap<usize, HashSet<usize>>;

fn load_graph() -> (EdgeSet, Graph) {
    let args: Vec<_> = std::env::args().collect();
    let filename = &args[1];
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let (reader_tx, reader_rx) = std::sync::mpsc::channel();
    let (edge_tx, edge_rx) = std::sync::mpsc::channel();
    let (graph_tx, graph_rx) = std::sync::mpsc::channel();

    let reader_thread = thread::spawn(move || {
        for line in reader.lines() {
            let line = line.unwrap();
            reader_tx.send(line).unwrap();
        }
    });

    let bridge_thread = thread::spawn(move || {
        while let Ok(line) = reader_rx.recv() {
            if line.starts_with('#') {
                continue;
            }

            let mut splited = line.split_whitespace();
            let src: usize = splited.next().unwrap().parse().unwrap();
            let dst: usize = splited.next().unwrap().parse().unwrap();
            if src == dst {
                continue;
            }

            edge_tx.send((src, dst)).unwrap();
            graph_tx.send((src, dst)).unwrap();
        }
    });

    let edge_thread = thread::spawn(move || {
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        while let Ok((src, dst)) = edge_rx.recv() {
            let (src, dst) = (src.min(dst), src.max(dst));
            edges.insert((src, dst));
        }
        edges
    });

    let graph_thread = thread::spawn(move || {
        let mut graph: HashMap<usize, HashSet<usize>> = HashMap::new();
        while let Ok((src, dst)) = graph_rx.recv() {
            graph.entry(src).or_default().insert(dst);
            graph.entry(dst).or_default().insert(src);
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
    let start_time = time::Instant::now();
    let (edges, graph) = load_graph();
    let loaded_time = time::Instant::now();
    println!(
        "Loading data: {} seconds",
        (loaded_time - start_time).as_secs_f64()
    );

    let accumulate: usize = edges
        .par_iter()
        .map(|(src, dst)| {
            graph
                .get(src)
                .unwrap()
                .intersection(graph.get(dst).unwrap())
                .count()
        })
        .sum();
    let end_time = time::Instant::now();
    println!(
        "Computation : {} seconds",
        (end_time - loaded_time).as_secs_f64()
    );

    let result = accumulate / 3;
    println!("{result}");
}
