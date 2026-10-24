use sgpu::{QueryPool, QueueType, create_query_pool};

use super::TaskGraph;

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
    fn new(frames_in_flight: usize, query_count: u32) -> Self {
        Self {
            pools: (0..frames_in_flight).map(|_| create_query_pool(query_count)).collect(),
            query_count,
            written: vec![false; frames_in_flight],
        }
    }
}

impl<'g, R, D> TaskGraph<'g, R, D> {
    pub(crate) fn ensure_timing_pools(&mut self, node_count: usize) {
        if !self.node_timing_enabled || node_count == 0 {
            return;
        }

        let query_count = u32::try_from(node_count.checked_mul(2).expect("node count overflows usize")).expect("too many nodes for timestamp queries");

        if self.node_timing.as_ref().is_some_and(|timing| timing.query_count == query_count && timing.pools.len() == self.frames_in_flight) {
            return;
        }

        if self.node_timing.is_some() {
            sgpu::wait_idle();
        }

        self.node_timing = Some(NodeTimingState::new(self.frames_in_flight, query_count));
    }

    pub(crate) fn timing_pool(&self, slot: usize) -> Option<&QueryPool> {
        if !self.node_timing_enabled {
            return None;
        }

        self.node_timing.as_ref().and_then(|timing| timing.pools.get(slot))
    }

    pub(crate) fn mark_timing_written(&mut self, slot: usize) {
        if self.node_timing_enabled
            && let Some(timing) = &mut self.node_timing
            && let Some(written) = timing.written.get_mut(slot)
        {
            *written = true;
        }
    }
}
