pub(crate) mod alias;
pub(crate) mod schedule;
pub(crate) mod sync;
pub(crate) mod timeline;

use sgpu::{AccessType, Counter, QueueType};

use super::ErasedNode;
use schedule::Schedule;
use sync::SubmissionPlan;
use timeline::Timeline;

const QUEUES: [QueueType; 3] = [
    QueueType::Graphics,
    QueueType::Compute,
    QueueType::Transfer,
];

// Node + the resources it accesses
pub(crate) struct NodeDeclaration<'g, R, D> {
    pub(crate) built: Box<dyn ErasedNode<R, D> + 'g>,
    pub(crate) queue: QueueType,
    pub(crate) images: Vec<(u32, AccessType)>,
    pub(crate) buffers: Vec<(u32, AccessType)>,
}

pub(crate) struct CompiledGraph<'g, R, D> {
    pub(crate) timelines: Vec<Timeline>,
    pub(crate) schedule: Schedule,
    pub(crate) built: Vec<Box<dyn ErasedNode<R, D> + 'g>>,
    pub(crate) runtime: RuntimeInfo,
    pub(crate) submissions: SubmissionPlan,
}

pub(crate) struct RuntimeInfo {
    pub(crate) executed_once: bool,
    pub(crate) curr_frame: usize,
    pub(crate) per_frame_counters: Vec<[Option<Counter>; 3]>,
    pub(crate) frame_executed: Vec<bool>,
}

impl RuntimeInfo {
    pub(crate) fn new(frames_in_flight: usize) -> Self {
        Self {
            executed_once: false,
            curr_frame: 0,
            per_frame_counters: vec![[None, None, None]; frames_in_flight],
            frame_executed: vec![false; frames_in_flight],
        }
    }
}
