use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::thread;

fn load_graph() -> (HashSet<(usize, usize)>, HashMap<usize, HashSet<usize>>) {
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
            if line.starts_with("#") {
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
        return edges;
    });

    let graph_thread = thread::spawn(move || {
        let mut graph: HashMap<usize, HashSet<usize>> = HashMap::new();
        while let Ok((src, dst)) = graph_rx.recv() {
            if !graph.contains_key(&src) {
                graph.insert(src, HashSet::new());
            }
            if !graph.contains_key(&dst) {
                graph.insert(dst, HashSet::new());
            }
            graph.get_mut(&src).unwrap().insert(dst);
            graph.get_mut(&dst).unwrap().insert(src);
        }
        return graph;
    });

    reader_thread.join().unwrap();
    bridge_thread.join().unwrap();
    let edges = edge_thread.join().unwrap();
    let graph = graph_thread.join().unwrap();

    (edges, graph)
}

fn main() {
    let (edges, graph) = load_graph();

    let mut result = 0;

    for (src, dst) in edges.iter() {
        let local = graph
            .get(src)
            .unwrap()
            .intersection(graph.get(dst).unwrap())
            .count();
        result += local;
    }

    let result = result / 3;
    println!("{result}");
}
