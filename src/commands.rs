#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
}

impl QueueType {
    pub(crate) const QUEUE_TYPES: [QueueType; 3] = [
        QueueType::Graphics,
        QueueType::Transfer,
        QueueType::Compute,
    ];
}

pub struct CommandBuffer {}
