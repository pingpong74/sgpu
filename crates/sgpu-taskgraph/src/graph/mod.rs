mod compile;
mod execute;
mod timeline;

use sgpu::{AcquiredImage, Buffer, Counter, Image, QueueType};

use self::compile::{CompiledGraph, NodeDeclaration};
use crate::node::{ExecutionContext, Node, NodeWriter};
use crate::resource::Id;

/// A graph consisting of Node
/// Each Node points to some work on the GPU
/// The `TaskGraph` handles all syncronization
pub struct TaskGraph<'g, R, D> {
    images: Vec<Image>,
    buffers: Vec<Buffer>,
    entries: Vec<NodeEntry<'g, R, D>>,
    present_id: Option<Id<Image>>,
    compiled: Option<CompiledGraph<'g, R, D>>,
    executed_once: bool,
    prev_counters: [Option<Counter>; 3],
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    /// Create a new empty `TaskGraph`
    pub fn new() -> TaskGraph<'g, R, D> {
        TaskGraph {
            images: Vec::new(),
            buffers: Vec::new(),
            entries: Vec::new(),
            present_id: None,
            compiled: None,
            executed_once: false,
            prev_counters: [None; 3],
        }
    }

    /// Add an external Image
    pub fn add_image(&mut self, image: Image) -> Id<Image> {
        self.compiled = None;
        let id = Id::new(self.images.len() as u32);
        self.images.push(image);
        id
    }

    /// Add an external Buffer
    pub fn add_buffer(&mut self, buffer: Buffer) -> Id<Buffer> {
        self.compiled = None;
        let id = Id::new(self.buffers.len() as u32);
        self.buffers.push(buffer);
        id
    }

    /// Add a Node
    pub fn add_node<N: Node<R, D> + 'g>(&mut self, config: N::Config) {
        self.compiled = None;
        self.entries.push(NodeEntry {
            queue: N::QUEUE,
            build: Box::new(move |res, w| Box::new(N::declare(res, &config, w))),
        });
    }

    /// This transitions the given Image to Presentation Optimal Image Layout
    pub fn present_transition(&mut self, id: Id<Image>) {
        assert!((id.index as usize) < self.images.len(), "image id {id:?} does not belong to this graph (foreign or stale Id?)");
        self.compiled = None;
        self.present_id = Some(id);
    }

    /// This updates the internal `sgpu::Image`
    pub fn set_image(&mut self, id: Id<Image>, image: Image) { self.images[id.index as usize] = image; }

    /// This updates the internal `sgpu::Buffer`
    pub fn set_buffer(&mut self, id: Id<Buffer>, buffer: Buffer) { self.buffers[id.index as usize] = buffer; }

    /// Resets the execution of the graph, Used for resizing
    pub fn reset_steady_state(&mut self) {
        self.executed_once = false;
        self.prev_counters = [None; 3];
    }

    /// Builds the Task graph
    pub fn compile(&mut self, res: &R) {
        let declared = self
            .entries
            .iter()
            .map(|entry| {
                let mut w = NodeWriter::new(self.images.len(), self.buffers.len());
                NodeDeclaration {
                    built: (entry.build)(res, &mut w),
                    images: w.images,
                    buffers: w.buffers,
                }
            })
            .collect::<Vec<_>>();

        let mut timelines = self.build_timelines(&declared);
        let present = self.append_present(&mut timelines);
        let adj = self.adj_list(&timelines, present);
        let (batches, task_to_batch) = self.assign_batches(&mut timelines, &adj, present);
        let mut queue_batches = self.queue_batches(&timelines, &batches, &task_to_batch, present);
        self.derive_sync(&timelines, &task_to_batch, &mut queue_batches);

        self.compiled = Some(CompiledGraph {
            batches: queue_batches,
            built: declared.into_iter().map(|d| d.built).collect(),
            present: present.is_some(),
        });
    }

    /// Execute the `TaskGraph`
    pub fn execute(&mut self, res: &R, data: &D) -> Counter {
        if self.compiled.is_none() {
            self.compile(res);
        }
        self.run(None, data)
    }

    /// Execute the `TaskGraph`, but waits for the swapchain image to be accquired
    pub fn execute_with_swapchain(&mut self, res: &R, data: &D, acquired: &AcquiredImage) -> Counter {
        if self.compiled.is_none() {
            self.compile(res);
        }
        self.run(Some(acquired), data)
    }
}

trait ErasedNode<R, D> {
    fn execute(&self, data: &D, ctx: &mut ExecutionContext<'_>);
}

impl<R, D, N: Node<R, D>> ErasedNode<R, D> for N {
    fn execute(&self, data: &D, ctx: &mut ExecutionContext<'_>) { Node::execute(self, data, ctx) }
}

struct NodeEntry<'g, R, D> {
    queue: QueueType,
    build: Box<dyn Fn(&R, &mut NodeWriter) -> Box<dyn ErasedNode<R, D> + 'g> + 'g>,
}
