mod compile;
mod execute;
mod timeline;

use std::sync::atomic::{AtomicU32, Ordering};

use sgpu::{AcquiredImage, Buffer, Counter, Image, ImageDescription, QueueType, destroy_image};

use self::compile::{CompiledGraph, NodeDeclaration};
use crate::node::{ExecutionContext, Node, NodeWriter};
use crate::resource::{Id, ImageViewPool, InnerImage};

/// A graph consisting of Node
/// Each Node points to some work on the GPU
/// The `TaskGraph` handles all syncronization
pub struct TaskGraph<'g, R, D> {
    graph_id: u32,
    images: Vec<InnerImage>,
    buffers: Vec<Buffer>,
    entries: Vec<NodeEntry<'g, R, D>>,
    present_id: Option<Id<Image>>,
    compiled: Option<CompiledGraph<'g, R, D>>,
    views: ImageViewPool,
    executed_once: bool,
    prev_counters: [Option<Counter>; 3],
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    /// Create a new empty `TaskGraph`
    pub fn new() -> TaskGraph<'g, R, D> {
        TaskGraph {
            graph_id: next_graph_id(),
            images: Vec::new(),
            buffers: Vec::new(),
            entries: Vec::new(),
            present_id: None,
            compiled: None,
            views: ImageViewPool::default(),
            executed_once: false,
            prev_counters: [None; 3],
        }
    }

    /// Add an external Image
    pub fn add_image(&mut self, image: Image) -> Id<Image> {
        self.compiled = None;
        self.push_image(InnerImage::External(image))
    }

    /// Create an Image owned by the graph, kept at a fixed size and destroyed
    /// with the graph
    pub fn create_image(&mut self, desc: &ImageDescription) -> Id<Image> {
        self.compiled = None;
        self.push_image(InnerImage::Persistent {
            image_desc: *desc,
            image: sgpu::create_image(desc),
            resizable: false,
        })
    }

    /// Create an Image owned by the graph that is recreated by
    /// `TaskGraph::resize` and destroyed with the graph
    pub fn create_resizable_image(&mut self, desc: &ImageDescription) -> Id<Image> {
        self.compiled = None;
        self.push_image(InnerImage::Persistent {
            image_desc: *desc,
            image: sgpu::create_image(desc),
            resizable: true,
        })
    }

    /// Add an external Buffer
    pub fn add_buffer(&mut self, buffer: Buffer) -> Id<Buffer> {
        self.compiled = None;
        let id = Id::new(self.buffers.len() as u32, self.graph_id);
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
        self.get_image_index(id);
        self.compiled = None;
        self.present_id = Some(id);
    }

    /// Rebind an external image (e.g. the next swapchain image). Graph-owned
    /// images cannot be rebound.
    pub fn set_image(&mut self, id: Id<Image>, image: Image) {
        let index = self.get_image_index(id) as usize;
        match &mut self.images[index] {
            InnerImage::External(current) => *current = image,
            InnerImage::Persistent { .. } => panic!("image {id:?} is owned by the graph and cannot be rebound"),
        }
    }

    /// This updates the internal `sgpu::Buffer`
    pub fn set_buffer(&mut self, id: Id<Buffer>, buffer: Buffer) {
        let index = self.get_buffer_index(id) as usize;
        self.buffers[index] = buffer;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        for (i, slot) in self.images.iter_mut().enumerate() {
            if let InnerImage::Persistent {
                resizable: true,
                ..
            } = slot
            {
                self.views.destroy_slot(i as u32);
            }
        }
        self.views.drain();
        for slot in &mut self.images {
            if let InnerImage::Persistent {
                image_desc,
                image,
                resizable: true,
            } = slot
            {
                destroy_image(*image);
                image_desc.extent.width = width;
                image_desc.extent.height = height;
                *image = sgpu::create_image(image_desc);
            }
        }
        self.reset_steady_state();
    }

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
                let mut w = NodeWriter::new(self.graph_id, &self.images, &self.buffers, &mut self.views);
                let built = (entry.build)(res, &mut w);
                NodeDeclaration {
                    built,
                    images: w.image_accesses,
                    buffers: w.buffer_accesses,
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

    fn push_image(&mut self, image: InnerImage) -> Id<Image> {
        let id = Id::new(self.images.len() as u32, self.graph_id);
        self.images.push(image);
        id
    }

    fn get_image_index(&self, id: Id<Image>) -> u32 { id.check(self.graph_id, self.images.len()) }

    fn get_buffer_index(&self, id: Id<Buffer>) -> u32 { id.check(self.graph_id, self.buffers.len()) }
}

impl<'g, R, D> Drop for TaskGraph<'g, R, D> {
    fn drop(&mut self) {
        self.views.destory_all();
        for slot in &self.images {
            if let InnerImage::Persistent { image, .. } = slot {
                destroy_image(*image);
            }
        }
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

static NEXT_GRAPH_ID: AtomicU32 = AtomicU32::new(1);

pub(crate) fn next_graph_id() -> u32 { NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed) }
