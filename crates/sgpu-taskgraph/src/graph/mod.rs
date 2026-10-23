mod compile;
mod execute;

use std::sync::atomic::{AtomicU32, Ordering};

use self::compile::{CompiledGraph, NodeDeclaration, RuntimeInfo, schedule::Schedule};
use crate::node::{ExecutionContext, Node, NodeWriter};
use crate::resource::{Id, ImageViewPool, InnerBuffer, InnerImage};
use sgpu::{AcquiredImage, Buffer, BufferDescription, Counter, Image, ImageDescription, QueueType, create_buffer, create_image, destroy_buffer, destroy_image};

/// A graph consisting of Node
/// Each Node points to some work on the GPU
/// The `TaskGraph` handles all syncronization
pub struct TaskGraph<'g, R, D> {
    graph_id: u32,
    images: Vec<InnerImage>,
    buffers: Vec<InnerBuffer>,
    entries: Vec<NodeEntry<'g, R, D>>,
    present_node: Option<usize>,
    compiled: Option<CompiledGraph<'g, R, D>>,
    views: ImageViewPool,
    frames_in_flight: usize,
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    /// Create a new empty `TaskGraph`
    pub fn new() -> TaskGraph<'g, R, D> {
        TaskGraph {
            graph_id: next_graph_id(),
            images: Vec::new(),
            buffers: Vec::new(),
            entries: Vec::new(),
            present_node: None,
            compiled: None,
            views: ImageViewPool::default(),
            frames_in_flight: 0,
        }
    }

    pub fn set_frames_in_flight(&mut self, f: usize) {
        assert!(f >= 1, "frame slot count must be at least 1");
        self.frames_in_flight = f;
        self.invalidate();
    }

    /// Add an external Image
    pub fn add_image(&mut self, image: Image) -> Id<Image> {
        self.invalidate();
        self.push_image(InnerImage::External(image))
    }

    /// Add an external Buffer
    pub fn add_buffer(&mut self, buffer: Buffer) -> Id<Buffer> {
        self.invalidate();
        let id = Id::new(self.buffers.len() as u32, self.graph_id);
        self.buffers.push(InnerBuffer::External(buffer));
        id
    }

    /// Create persistent images owned by the graph.
    /// If `resize` is true, the image follows `resize()` and its contents are discarded.
    pub fn create_persistent_image(&mut self, desc: &ImageDescription, resize: bool) -> Id<Image> {
        self.invalidate();
        self.push_image(InnerImage::Persistent {
            image_desc: *desc,
            images: Vec::new(),
            resize,
        })
    }

    /// Create persistent buffers owned by the graph
    /// Not used for aliasing
    pub fn create_persistent_buffer(&mut self, desc: &BufferDescription) -> Id<Buffer> {
        self.invalidate();
        let id = Id::new(self.buffers.len() as u32, self.graph_id);
        self.buffers.push(InnerBuffer::Persistent {
            buffer_desc: *desc,
            buffers: Vec::new(),
        });
        id
    }

    /// Create a transient image owned by the graph
    /// Used for aliasing
    /// resize: the image's extent tracks screen size on resize()
    pub fn create_transient_image(&mut self, desc: &ImageDescription, resize: bool) -> Id<Image> {
        self.invalidate();
        self.push_image(InnerImage::Transient {
            image_desc: *desc,
            images: Vec::new(),
            resize,
        })
    }

    /// Create a transient buffer owned by the graph
    /// Used for aliasing
    pub fn create_transient_buffer(&mut self, desc: &BufferDescription) -> Id<Buffer> {
        self.invalidate();
        let id = Id::new(self.buffers.len() as u32, self.graph_id);
        self.buffers.push(InnerBuffer::Transient {
            buffer_desc: *desc,
            buffers: Vec::new(),
        });
        id
    }

    /// Add a Node
    pub fn add_node<N: Node<R, D> + 'g>(&mut self, config: N::Config) {
        self.invalidate();
        self.entries.push(NodeEntry {
            queue: N::QUEUE,
            build: Box::new(move |res, w| Box::new(N::declare(res, &config, w))),
        });
    }

    /// This marks the external image the app presents with
    pub fn present_target(&mut self, id: Id<Image>) {
        let index = self.get_image_index(id) as usize;
        if self.images[index].is_graph_owned() {
            panic!("only external images may be present targets ({id:?})");
        }

        assert!(self.present_node.is_none(), "the graph already has a present target");
        assert!(self.compiled.is_none(), "present target must be marked before compiling; rebuild the graph or mark it at declare time");

        self.present_node = Some(self.entries.len());
        self.entries.push(NodeEntry {
            queue: <crate::node::PresentNode as Node<R, D>>::QUEUE,
            build: Box::new(move |res, w| Box::new(<crate::node::PresentNode as Node<R, D>>::declare(res, &id, w))),
        });
    }

    /// Rebind an external image
    pub fn set_image(&mut self, id: Id<Image>, image: Image) {
        let index = self.get_image_index(id) as usize;
        match self.images[index] {
            InnerImage::External(_) => self.images[index] = InnerImage::External(image),
            _ => panic!("image {id:?} is not a plain external image and cannot be rebound"),
        }
    }

    /// Rebind an external buffer
    pub fn set_buffer(&mut self, id: Id<Buffer>, buffer: Buffer) {
        let index = self.get_buffer_index(id) as usize;
        match self.buffers[index] {
            InnerBuffer::External(_) => self.buffers[index] = InnerBuffer::External(buffer),
            _ => panic!("buffer {id:?} is not a plain external buffer and cannot be rebound"),
        }
    }

    /// Resize the resizable images and rebuild the affected resources and plan.
    pub fn resize(&mut self, width: u32, height: u32) {
        assert!(self.frames_in_flight > 0, "set_frames_in_flight must be called before resize");
        sgpu::wait_idle();

        for img in &mut self.images {
            match img {
                InnerImage::Persistent {
                    image_desc,
                    resize: true,
                    ..
                }
                | InnerImage::Transient {
                    image_desc,
                    resize: true,
                    ..
                } => {
                    image_desc.extent.width = width;
                    image_desc.extent.height = height;
                }
                _ => {}
            }
        }

        self.materialize_persistent_images(true);

        let (alias, submissions) = match &self.compiled {
            Some(compiled) => {
                let alias = self.build_alias_plan(&compiled.timelines, &compiled.schedule);
                let submissions = self.build_submission_plan(&compiled.timelines, &compiled.schedule, &alias);
                (alias, submissions)
            }
            None => return,
        };

        let mut compiled = self.compiled.take().expect("compiled graph checked above");
        self.materialize_transients(&alias);
        compiled.submissions = submissions;
        compiled.runtime = RuntimeInfo::new(self.frames_in_flight);
        self.compiled = Some(compiled);
    }

    /// Builds the task graph.
    pub fn compile(&mut self, res: &R) {
        assert!(self.frames_in_flight > 0, "set_frames_in_flight must be called before compile");

        let declared = self
            .entries
            .iter()
            .map(|entry| {
                let mut w = NodeWriter::new(self.graph_id, &self.images, &self.buffers, &mut self.views);
                let built = (entry.build)(res, &mut w);
                NodeDeclaration {
                    built,
                    queue: entry.queue,
                    images: w.image_accesses,
                    buffers: w.buffer_accesses,
                }
            })
            .collect::<Vec<_>>();

        let timelines = self.build_timelines(&declared);
        for tl in &timelines {
            tl.validate_present();
        }
        let node_queues: Vec<QueueType> = declared.iter().map(|declaration| declaration.queue).collect();
        let schedule = Schedule::build(&timelines, &node_queues);
        let alias = self.build_alias_plan(&timelines, &schedule);
        let submissions = self.build_submission_plan(&timelines, &schedule, &alias);

        sgpu::wait_idle();
        self.materialize_persistent_images(false);
        self.materialize_persistent_buffers();
        self.materialize_transients(&alias);

        self.compiled = Some(CompiledGraph {
            timelines,
            schedule,
            built: declared.into_iter().map(|d| d.built).collect(),
            runtime: RuntimeInfo::new(self.frames_in_flight),
            submissions,
        });
    }

    /// Execute this taskgraph, requires self.compile first
    pub fn execute(&mut self, data: &D) -> Counter { self.run(None, data) }

    /// Execute this taskgraph on the swapchain's frame timeline
    pub fn execute_with_swapchain(&mut self, data: &D, acquired: &AcquiredImage) -> Counter { self.run(Some(acquired), data) }

    fn materialize_persistent_images(&mut self, recreate_resize: bool) {
        let slots = self.frames_in_flight;

        let mut old_images: Vec<(usize, Vec<Image>)> = Vec::new();
        for (index, rack) in self.images.iter_mut().enumerate() {
            if let InnerImage::Persistent {
                images,
                resize,
                ..
            } = rack
                && (images.len() != slots || (recreate_resize && *resize))
            {
                old_images.push((index, std::mem::take(images)));
            }
        }

        for (index, images) in old_images {
            if images.is_empty() {
                continue;
            }
            self.views.destroy_slot(index as u32);
            for image in images {
                destroy_image(image);
            }
        }

        for rack in &mut self.images {
            if let InnerImage::Persistent {
                image_desc,
                images,
                ..
            } = rack
                && images.is_empty()
            {
                *images = (0..slots).map(|_| create_image(image_desc)).collect();
            }
        }
    }

    fn materialize_persistent_buffers(&mut self) {
        let slots = self.frames_in_flight;

        let mut old_buffers: Vec<Vec<Buffer>> = Vec::new();
        for buffer in &mut self.buffers {
            if let InnerBuffer::Persistent {
                buffers, ..
            } = buffer
                && buffers.len() != slots
            {
                old_buffers.push(std::mem::take(buffers));
            }
        }

        for buffers in old_buffers {
            for buffer in buffers {
                destroy_buffer(buffer);
            }
        }

        for rack in &mut self.buffers {
            if let InnerBuffer::Persistent {
                buffer_desc,
                buffers,
            } = rack
                && buffers.is_empty()
            {
                *buffers = (0..slots).map(|_| create_buffer(buffer_desc)).collect();
            }
        }
    }

    fn invalidate(&mut self) { self.compiled = None; }

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
        sgpu::wait_idle();
        self.views.destroy_all();
        for rack in &self.images {
            match rack {
                InnerImage::Persistent { images, .. } | InnerImage::Transient { images, .. } => {
                    images.iter().for_each(|img| destroy_image(*img));
                }
                _ => {}
            }
        }
        for rack in self.buffers.iter() {
            match rack {
                InnerBuffer::Persistent {
                    buffers, ..
                }
                | InnerBuffer::Transient {
                    buffers, ..
                } => {
                    buffers.iter().for_each(|buffer| destroy_buffer(*buffer));
                }
                _ => {}
            }
        }
    }
}

pub(crate) trait ErasedNode<R, D> {
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
