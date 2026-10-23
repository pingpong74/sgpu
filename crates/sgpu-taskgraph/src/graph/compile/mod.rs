pub(crate) mod alias;
pub(crate) mod schedule;
pub(crate) mod sync;
pub(crate) mod timeline;

use sgpu::{AccessType, Counter, QueueType};

use super::ErasedNode;
use schedule::Schedule;
use sync::CompiledSync;
use timeline::Timeline;

const QUEUES: [QueueType; 3] = [
    QueueType::Graphics,
    QueueType::Compute,
    QueueType::Transfer,
];

// Node + the resources it accesses
pub(crate) struct NodeDeclaration<'g, R, D> {
    pub(crate) built: Box<dyn ErasedNode<R, D> + 'g>,
    pub(crate) images: Vec<(u32, AccessType)>,
    pub(crate) buffers: Vec<(u32, AccessType)>,
}

/// The compiled graph: structure, built nodes, runtime state, and sync plan.
pub(crate) struct CompiledGraph<'g, R, D> {
    pub(crate) timelines: Vec<Timeline>,
    pub(crate) schedule: Schedule,
    pub(crate) built: Vec<Box<dyn ErasedNode<R, D> + 'g>>,
    /// Per-slot submission counters and executed flags. Reset when resources
    /// or the slot count change.
    pub(crate) runtime: RuntimeInfo,
    pub(crate) sync: CompiledSync,
}

pub(crate) struct RuntimeInfo {
    pub(crate) executed_once: bool,
    pub(crate) curr_frame: usize,
    /// Per-slot ring of each queue's last counter from the last frame that
    /// ran in that slot. Cross-cycle edges wait on this; the frame loop
    /// seeds its rolling queue counters from the previous frame's rotation
    /// position. A None means the queue never submitted in that cycle.
    pub(crate) per_frame_counters: Vec<[Option<Counter>; 3]>,
    // Whether the frame has been executed once
    pub(crate) frame_executed: Vec<bool>,
}

impl RuntimeInfo {
    /// Fresh books for a graph instanced for `slot_count` frames: nothing
    /// has run, no previous-cycle counters exist anywhere.
    pub(crate) fn new(frames_in_flight: usize) -> Self {
        Self {
            executed_once: false,
            curr_frame: 0,
            per_frame_counters: vec![[None, None, None]; frames_in_flight],
            frame_executed: vec![false; frames_in_flight],
        }
    }
}
