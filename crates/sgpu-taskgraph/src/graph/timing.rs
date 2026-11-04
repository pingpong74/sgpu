use sgpu::{QueryPool, QueueType, create_query_pool};

/// Per-node GPU timings for one executed frame.
#[derive(Clone, Debug)]
pub struct FrameTimings {
    /// The frame slot the timings were recorded in.
    pub slot: usize,
    /// One entry per user node, in graph order. The internal present marker is excluded.
    pub nodes: Vec<NodeTiming>,
}

/// GPU time spent inside one node.
#[derive(Clone, Debug)]
pub struct NodeTiming {
    /// Node index in graph order.
    pub node: usize,
    /// Node type name.
    pub label: &'static str,
    /// Queue the node was scheduled on.
    pub queue: QueueType,
    /// Timestamp before the node, in nanoseconds.
    pub start_ns: f64,
    /// Timestamp after the node, in nanoseconds.
    pub end_ns: f64,
    /// `end_ns - start_ns`.
    pub duration_ns: f64,
}

pub(crate) struct NodeTimingState {
    pub(crate) pools: Vec<QueryPool>,
    pub(crate) query_count: u32,
    pub(crate) written: Vec<bool>,
}

impl NodeTimingState {
    pub(crate) fn new(frames_in_flight: usize, query_count: u32) -> Self {
        Self {
            pools: (0..frames_in_flight).map(|_| create_query_pool(query_count)).collect(),
            query_count,
            written: vec![false; frames_in_flight],
        }
    }
}
