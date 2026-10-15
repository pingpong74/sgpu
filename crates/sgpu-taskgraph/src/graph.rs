use std::ops::Range;

use sgpu::{AccessType, AcquiredImage, Buffer, CommandBuffer, Counter, GlobalBarrier, Image, ImageBarrier, ImageLayout, PipelineStage, QueueType};
use smallvec::SmallVec;

use crate::node::{ExecutionContext, Node, NodeWriter};
use crate::resource::Id;

/// Trait-object wrapper so heterogeneous nodes can be stored together. One
/// virtual call per node per frame — negligible next to the GPU work itself.
trait ErasedNode<R, D> {
    fn execute(&self, data: &D, ctx: &mut ExecutionContext<'_>);
}

impl<R, D, N: Node<R, D>> ErasedNode<R, D> for N {
    fn execute(&self, data: &D, ctx: &mut ExecutionContext<'_>) { Node::execute(self, data, ctx) }
}

/// Factory that (re)builds one node at compile time: `N::declare(res, &config, w)`
/// with the config captured by reference.
type NodeFactory<'g, R, D> = Box<dyn Fn(&R, &mut NodeWriter) -> Box<dyn ErasedNode<R, D> + 'g> + 'g>;

/// A registered node: its queue plus a factory that (re)builds it at compile
/// time. `declare` re-runs on every recompile, so it must be pure — the config
/// is captured here by reference and reused for every build.
struct NodeEntry<'g, R, D> {
    queue: QueueType,
    build: NodeFactory<'g, R, D>,
}

/// A pre-synthesized image barrier, ready to emit each frame. References the
/// graph's image by index — the view is resolved at execute, so a compiled
/// graph never holds a stale handle.
struct CompiledImageBarrier {
    image: u32,
    prev: AccessType,
    next: AccessType,
    discard: bool,
}

/// Merged buffer-hazard barriers for one node (folded into a single global).
struct CompiledGlobal {
    prev: Vec<AccessType>,
    next: Vec<AccessType>,
}

/// One node's synthesized synchronization, for one chain flavor.
struct NodeBarriers {
    image_barriers: Vec<CompiledImageBarrier>,
    global: Option<CompiledGlobal>,
    /// Cross-queue waits: (source queue, destination stage).
    waits: Vec<(QueueType, PipelineStage)>,
}

struct CompiledNode {
    queue: QueueType,
    /// First-ever execute: chains start from `None` (resources are fresh —
    /// first writes discard their undefined contents).
    cold: NodeBarriers,
    /// Every execute after: chains are seeded with the walk's final state, so
    /// the first access of the frame is ordered against the *previous* frame's
    /// last access to the same resource (wrap-around WAR/WAW).
    hot: NodeBarriers,
}

struct CompiledGraph<'g, R, D> {
    nodes: Vec<CompiledNode>,
    /// The built nodes themselves, parallel to `nodes`.
    built: Vec<Box<dyn ErasedNode<R, D> + 'g>>,
}

type ResourceState = Option<(AccessType, QueueType)>;

/// Walks the declared accesses in node order and synthesizes per-node
/// synchronization, starting from the given per-resource states.
///
/// - Same queue: hazards become barriers (images: one `ImageBarrier` per
///   resource; buffers: folded into one `GlobalBarrier`).
/// - Cross-queue: a semaphore wait on the source queue's submit counter; a
///   barrier is emitted too if the image layout changes.
/// - First access overall (state `None`): a write discards contents (transition
///   from `UNDEFINED`); a read needs nothing.
///
/// Returns the barriers per node and the final per-resource states.
#[allow(clippy::too_many_arguments)]
fn synthesize(
    queues: &[QueueType],
    image_accesses: &[(u32, AccessType)],
    buffer_accesses: &[(u32, AccessType)],
    image_ranges: &[Range<usize>],
    buffer_ranges: &[Range<usize>],
    mut last_image: Vec<ResourceState>,
    mut last_buffer: Vec<ResourceState>,
    is_hot: bool,
) -> (Vec<NodeBarriers>, Vec<ResourceState>, Vec<ResourceState>) {
    let mut nodes = Vec::with_capacity(queues.len());

    for (i, &queue) in queues.iter().enumerate() {
        let mut image_barriers: Vec<CompiledImageBarrier> = Vec::new();
        let mut global_prev: Vec<AccessType> = Vec::new();
        let mut global_next: Vec<AccessType> = Vec::new();
        let mut waits: Vec<(QueueType, PipelineStage)> = Vec::new();

        for &(idx, a) in &image_accesses[image_ranges[i].clone()] {
            let idx = idx as usize;
            match last_image[idx] {
                None => {
                    if a.is_write() {
                        image_barriers.push(CompiledImageBarrier {
                            image: idx as u32,
                            prev: AccessType::None,
                            next: a,
                            discard: true,
                        });
                    }
                }
                Some((prev, prev_q)) => {
                    let layout_change = a.layout_differs(prev);
                    let needs_sync = prev.is_write() || a.is_write() || layout_change;
                    if needs_sync {
                        if prev_q != queue {
                            // Cross-queue: the semaphore wait provides the
                            // execution + memory dependency; a barrier is only
                            // needed to actually change the layout.
                            waits.push((prev_q, a.stage()));
                            if layout_change {
                                let discard = is_hot && prev == AccessType::Present && a.is_write();
                                image_barriers.push(CompiledImageBarrier {
                                    image: idx as u32,
                                    prev,
                                    next: a,
                                    discard,
                                });
                            }
                        } else {
                            let discard = is_hot && prev == AccessType::Present && a.is_write();
                            image_barriers.push(CompiledImageBarrier {
                                image: idx as u32,
                                prev,
                                next: a,
                                discard,
                            });
                        }
                    }
                }
            }
            last_image[idx] = Some((a, queue));
        }

        for &(idx, a) in &buffer_accesses[buffer_ranges[i].clone()] {
            let idx = idx as usize;
            if let Some((prev, prev_q)) = last_buffer[idx] {
                let needs_sync = prev.is_write() || a.is_write();
                if needs_sync {
                    if prev_q != queue {
                        waits.push((prev_q, a.stage()));
                    } else {
                        global_prev.push(prev);
                        global_next.push(a);
                    }
                }
            }
            last_buffer[idx] = Some((a, queue));
        }

        let global = if global_prev.is_empty() {
            None
        } else {
            Some(CompiledGlobal {
                prev: global_prev,
                next: global_next,
            })
        };

        nodes.push(NodeBarriers {
            image_barriers,
            global,
            waits,
        });
    }

    (nodes, last_image, last_buffer)
}

/// The task graph. Add **external** resources + nodes, then [`Graph::execute`]
/// (or [`Graph::execute_with_swapchain`]) every frame.
///
/// - `R` — your resources struct: the bundle of [`Id`] handles (plus pipelines
///   and friends) handed to every node's `declare` and to [`Graph::compile`] /
///   [`Graph::execute`].
/// - `D` — per-frame data: one value passed straight through to every node's
///   `execute`. It never enters the graph.
///
/// Execution order == add order. Barriers and cross-queue waits are synthesized
/// from the declared accesses at compile time — once, then replayed every frame
/// with no per-frame allocation (flat vectors + an inline `SmallVec` for
/// barriers), one command buffer per queue run, and one submit per queue run.
/// The first execute uses "cold" chains (resources fresh); every execute after
/// uses "hot" chains seeded with the previous frame's final state, which orders
/// wrap-around (cross-frame) hazards.
pub struct Graph<'g, R, D> {
    images: Vec<Image>,
    /// User-owned external buffers. See [`Graph::images`].
    buffers: Vec<Buffer>,
    entries: Vec<NodeEntry<'g, R, D>>,
    compiled: Option<CompiledGraph<'g, R, D>>,
    /// Whether at least one execute has run since the last
    /// [`Graph::reset_steady_state`] — selects hot vs cold chains.
    executed_once: bool,
    /// Each queue's last submit counter from the previous execute. Hot
    /// cross-queue wrap-around waits reference these (the source queue hasn't
    /// run yet this frame).
    prev_counters: [Option<Counter>; 3],
}

impl<'g, R: 'g, D: 'g> Default for Graph<'g, R, D> {
    fn default() -> Self { Self::new() }
}

impl<'g, R: 'g, D: 'g> Graph<'g, R, D> {
    pub fn new() -> Graph<'g, R, D> {
        Graph {
            images: Vec::new(),
            buffers: Vec::new(),
            entries: Vec::new(),
            compiled: None,
            executed_once: false,
            prev_counters: [None; 3],
        }
    }

    /// Add an existing, user-owned image. The graph tracks its accesses but
    /// never creates or destroys it.
    pub fn add_image(&mut self, image: Image) -> Id<Image> {
        self.compiled = None;
        let id = Id::new(self.images.len() as u32);
        self.images.push(image);
        id
    }

    /// Add an existing, user-owned buffer. See [`Graph::add_image`].
    pub fn add_buffer(&mut self, buffer: Buffer) -> Id<Buffer> {
        self.compiled = None;
        let id = Id::new(self.buffers.len() as u32);
        self.buffers.push(buffer);
        id
    }

    /// Add a node. Nodes execute in the order they were added. The `config` is
    /// handed to `N::declare` at (re)compile time, which builds the node.
    ///
    /// Rust can't always infer `N` from `N::Config` alone — use the turbofish
    /// when needed: `graph.add_node::<MyNode>(config)`.
    pub fn add_node<N: Node<R, D> + 'g>(&mut self, config: N::Config)
    where
        N::Config: 'g,
    {
        self.compiled = None;
        self.entries.push(NodeEntry {
            queue: N::QUEUE,
            build: Box::new(move |res, w| Box::new(N::declare(res, &config, w))),
        });
    }

    /// Rebind the image behind `id` without invalidating the compiled graph —
    /// e.g. the next swapchain image each frame. Barriers hold indices and
    /// views resolve at execute, so nothing is recompiled.
    pub fn set_image(&mut self, id: Id<Image>, image: Image) { self.images[id.index as usize] = image; }

    /// See [`Graph::set_image`].
    pub fn set_buffer(&mut self, id: Id<Buffer>, buffer: Buffer) { self.buffers[id.index as usize] = buffer; }

    /// Forget the frame-to-frame state (hot chains + cross-queue counters) and
    /// return to cold chains. Call this when the steady state broke: external
    /// resources were recreated (resize), or the GPU was idled — e.g.
    /// `Swapchain::resize` calls `wait_idle`, so the next execute's cold chains
    /// are correct again for the fresh `UNDEFINED` images.
    pub fn reset_steady_state(&mut self) {
        self.executed_once = false;
        self.prev_counters = [None; 3];
    }

    /// (Re)build the nodes and synthesize both barrier chains. Runs
    /// automatically on execute if the graph is dirty (an `add_*` since the last
    /// compile); call it explicitly to force a rebuild.
    pub fn compile(&mut self, res: &R) {
        let image_count = self.images.len();
        let buffer_count = self.buffers.len();

        // 1. Build every node via its factory, collecting declared accesses.
        let mut image_accesses: Vec<(u32, AccessType)> = Vec::new();
        let mut buffer_accesses: Vec<(u32, AccessType)> = Vec::new();
        let mut image_ranges: Vec<Range<usize>> = Vec::with_capacity(self.entries.len());
        let mut buffer_ranges: Vec<Range<usize>> = Vec::with_capacity(self.entries.len());
        let mut built: Vec<Box<dyn ErasedNode<R, D> + 'g>> = Vec::with_capacity(self.entries.len());
        let mut queues: Vec<QueueType> = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            let i0 = image_accesses.len();
            let b0 = buffer_accesses.len();
            let mut w = NodeWriter::new(image_count, buffer_count);
            built.push((entry.build)(res, &mut w));
            image_accesses.extend(w.images);
            buffer_accesses.extend(w.buffers);
            image_ranges.push(i0..image_accesses.len());
            buffer_ranges.push(b0..buffer_accesses.len());
            queues.push(entry.queue);
        }

        // 2. Cold chain: everything starts fresh.
        let (cold_nodes, final_images, final_buffers) = synthesize(&queues, &image_accesses, &buffer_accesses, &image_ranges, &buffer_ranges, vec![None; image_count], vec![None; buffer_count], false);

        // 3. Hot chain: seeded with the cold walk's final state — except
        //    resources whose final access is `Present`. Presentable images are
        //    recycled by the swapchain (a given slot may even be used for the
        //    first time), so the hot chain always begins them with a fresh
        //    `UNDEFINED` discard transition, which is legal for every slot —
        //    and cross-frame ordering comes from the acquire semaphore.
        let hot_images: Vec<ResourceState> = final_images
            .iter()
            .map(|s| match s {
                Some((AccessType::Present, _)) => None,
                other => *other,
            })
            .collect();
        let (hot_nodes, _, _) = synthesize(&queues, &image_accesses, &buffer_accesses, &image_ranges, &buffer_ranges, hot_images, final_buffers, true);

        let nodes = cold_nodes
            .into_iter()
            .zip(hot_nodes)
            .enumerate()
            .map(|(i, (cold, hot))| CompiledNode {
                queue: queues[i],
                cold,
                hot,
            })
            .collect();

        self.compiled = Some(CompiledGraph {
            nodes,
            built,
        });
    }

    /// Run the whole graph once. Auto-compiles with `res` if dirty, then
    /// records + submits one command buffer per queue run and returns the final
    /// submit's [`Counter`].
    pub fn execute(&mut self, res: &R, data: &D) -> Counter {
        if self.compiled.is_none() {
            self.compile(res);
        }
        self.run(None, data)
    }

    /// [`Graph::execute`] for swapchain frames: additionally waits the acquired
    /// image's acquire semaphore on the first command buffer of this run.
    /// Rebind the next image first via [`Graph::set_image`], and present the
    /// returned [`Counter`] outside the graph.
    pub fn execute_with_swapchain(&mut self, res: &R, data: &D, acquired: &AcquiredImage) -> Counter {
        if self.compiled.is_none() {
            self.compile(res);
        }
        self.run(Some(acquired), data)
    }

    fn run(&mut self, acquired: Option<&AcquiredImage>, data: &D) -> Counter {
        let Graph {
            images,
            buffers,
            entries: _,
            compiled,
            executed_once,
            prev_counters,
        } = self;
        let compiled = compiled.as_ref().expect("graph compiled before run");

        let hot = *executed_once;
        let mut last_counter = *prev_counters;
        let mut current_queue: Option<QueueType> = None;
        let mut current_cmd: Option<CommandBuffer> = None;
        let mut final_counter: Option<Counter> = None;

        for (built, cnode) in compiled.built.iter().zip(compiled.nodes.iter()) {
            if current_queue != Some(cnode.queue) {
                if let Some(cmd) = current_cmd.take() {
                    let c = sgpu::submit(&[cmd]);
                    if let Some(q) = current_queue {
                        last_counter[q as usize] = Some(c);
                    }
                    final_counter = Some(c);
                }
                let mut cmd = sgpu::record(cnode.queue);
                // The acquire semaphore is per-frame: the first submission of
                // the run waits it, and only that submission (binary semaphore).
                if let (None, Some(acquired)) = (current_queue, acquired) {
                    cmd.wait_for_swapchain_image(acquired);
                }
                current_cmd = Some(cmd);
                current_queue = Some(cnode.queue);
            }

            let cmd = current_cmd.as_mut().expect("command buffer for current queue run");
            let barriers = if hot {
                &cnode.hot
            } else {
                &cnode.cold
            };

            // Cross-queue waits — on this frame's counters where the source
            // queue already ran, else on the previous frame's (wrap-around).
            for &(src_q, stage) in &barriers.waits {
                if let Some(c) = last_counter[src_q as usize] {
                    cmd.wait_for(c, stage);
                }
            }

            // Synthesized barriers: batched into a single vkCmdPipelineBarrier2.
            let global = barriers.global.as_ref().map(|g| GlobalBarrier {
                previous_accesses: &g.prev,
                next_accesses: &g.next,
            });

            if global.is_some() || !barriers.image_barriers.is_empty() {
                let mut imgs: SmallVec<[ImageBarrier; 8]> = SmallVec::new();
                for cb in &barriers.image_barriers {
                    imgs.push(ImageBarrier {
                        view: images[cb.image as usize].default_view(),
                        previous_accesses: std::slice::from_ref(&cb.prev),
                        next_accesses: std::slice::from_ref(&cb.next),
                        previous_layout: ImageLayout::Optimal,
                        next_layout: ImageLayout::Optimal,
                        discard_contents: cb.discard,
                    });
                }
                cmd.barriers(global.as_ref(), &imgs);
            }

            // Hand the pass its typed context and record.
            let mut ctx = ExecutionContext {
                cmd,
                images,
                buffers,
            };
            built.execute(data, &mut ctx);
        }

        // Flush the final run.
        if let Some(cmd) = current_cmd.take() {
            final_counter = Some(sgpu::submit(&[cmd]));
        }

        *prev_counters = last_counter;
        *executed_once = true;

        final_counter.expect("graph has no nodes — add at least one node")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{TransferRead, TransferWrite};
    use crate::{PassBuffer, PassImage};
    use sgpu::*;

    fn init() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| sgpu_init(&SgpuInititizationInfo::default()));
    }

    /// The graph's resources struct: the bundle handed to every node + compile.
    struct Res {
        buf: Id<Buffer>,
    }

    /// Per-node build config (differs per Fill instance — neither `Res` nor the
    /// per-frame data can carry it).
    struct FillConfig {
        value: u32,
        size: u64,
    }

    macro_rules! fill_node {
        ($name:ident, $queue:expr) => {
            struct $name {
                buf: PassBuffer<TransferWrite>,
                value: u32,
                size: u64,
            }
            impl Node<Res, ()> for $name {
                const QUEUE: QueueType = $queue;
                type Config = FillConfig;
                fn declare(res: &Res, config: &FillConfig, w: &mut NodeWriter) -> Self {
                    $name {
                        buf: w.buffer::<TransferWrite>(res.buf),
                        value: config.value,
                        size: config.size,
                    }
                }
                fn execute(&self, _: &(), ctx: &mut ExecutionContext<'_>) {
                    let b = ctx.buffer(self.buf);
                    ctx.cmd().fill_buffer(&b, 0, self.size, self.value);
                }
            }
        };
    }
    fill_node!(FillGraphics, QueueType::Graphics);
    fill_node!(FillTransfer, QueueType::Transfer);

    fn make_buffer() -> Buffer {
        create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE,
            size: 4 * 16,
            memory_type: MemoryType::HostVisible,
        })
    }

    #[test]
    fn same_queue_waw_barrier() {
        init();
        let buffer = make_buffer();

        let mut g: Graph<Res, ()> = Graph::new();
        let res = Res {
            buf: g.add_buffer(buffer),
        };
        g.add_node::<FillGraphics>(FillConfig {
            value: 0xAA,
            size: 64,
        });
        g.add_node::<FillGraphics>(FillConfig {
            value: 0xBB,
            size: 64,
        });

        wait(g.execute(&res, &()));

        {
            let data = buffer_as_slice::<u32>(&buffer);
            assert!(data.iter().all(|&x| x == 0xBB), "expected 0xBB");
        }

        destroy_buffer(buffer);
    }

    #[test]
    fn cross_queue_semaphore_sync() {
        init();
        let buffer = make_buffer();

        let mut g: Graph<Res, ()> = Graph::new();
        let res = Res {
            buf: g.add_buffer(buffer),
        };
        // Transfer writes first, graphics writes second — the graph must insert
        // a cross-queue semaphore wait or the fills race.
        g.add_node::<FillTransfer>(FillConfig {
            value: 0xAA,
            size: 64,
        });
        g.add_node::<FillGraphics>(FillConfig {
            value: 0xBB,
            size: 64,
        });

        wait(g.execute(&res, &()));

        {
            let data = buffer_as_slice::<u32>(&buffer);
            assert!(data.iter().all(|&x| x == 0xBB), "expected 0xBB");
        }

        destroy_buffer(buffer);
    }

    #[test]
    fn image_layout_transitions_cross_queue() {
        init();

        const W: u32 = 4;
        const H: u32 = 4;
        const BYTES: u64 = (W * H * 4) as u64; // Rgba8Unorm

        let upload = create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST,
            size: BYTES,
            memory_type: MemoryType::HostVisible,
        });
        let readback = create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST,
            size: BYTES,
            memory_type: MemoryType::HostVisible,
        });
        {
            let bytes = buffer_as_slice_mut::<u8>(&upload);
            bytes.fill(0x11);
        }

        let img = create_image(&ImageDescription {
            usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST,
            format: Format::Rgba8Unorm,
            image_type: ImageType::Type2D,
            extent: Extent3D {
                width: W,
                height: H,
                depth: 1,
            },
            memory_type: MemoryType::DeviceLocal,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Type1,
            ..Default::default()
        });

        let region = BufferImageCopyRegion {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: ImageSubresources::default(),
            image_offset: Offset3D {
                x: 0,
                y: 0,
                z: 0,
            },
            image_extent: Extent3D {
                width: W,
                height: H,
                depth: 1,
            },
        };

        struct FrameRes {
            img: Id<Image>,
            readback: Id<Buffer>,
        }

        struct UploadConfig {
            upload: Buffer,
            region: BufferImageCopyRegion,
        }

        struct Upload {
            img: PassImage<TransferWrite>,
            upload: Buffer,
            region: BufferImageCopyRegion,
        }
        impl Node<FrameRes, ()> for Upload {
            const QUEUE: QueueType = QueueType::Transfer;
            type Config = UploadConfig;
            fn declare(res: &FrameRes, config: &UploadConfig, w: &mut NodeWriter) -> Self {
                Upload {
                    img: w.image::<TransferWrite>(res.img),
                    upload: config.upload,
                    region: config.region,
                }
            }
            fn execute(&self, _: &(), ctx: &mut ExecutionContext<'_>) {
                let dst = ctx.image(self.img);
                ctx.cmd().copy_buffer_to_image(&self.upload, &dst, &[self.region]);
            }
        }

        struct Readback {
            img: PassImage<TransferRead>,
            readback: PassBuffer<TransferWrite>,
            region: BufferImageCopyRegion,
        }
        impl Node<FrameRes, ()> for Readback {
            const QUEUE: QueueType = QueueType::Graphics;
            type Config = BufferImageCopyRegion;
            fn declare(res: &FrameRes, region: &BufferImageCopyRegion, w: &mut NodeWriter) -> Self {
                Readback {
                    img: w.image::<TransferRead>(res.img),
                    readback: w.buffer::<TransferWrite>(res.readback),
                    region: *region,
                }
            }
            fn execute(&self, _: &(), ctx: &mut ExecutionContext<'_>) {
                let src = ctx.image(self.img);
                let dst = ctx.buffer(self.readback);
                ctx.cmd().copy_image_to_buffer(&src, &dst, &[self.region]);
            }
        }

        let mut g: Graph<FrameRes, ()> = Graph::new();
        let res = FrameRes {
            img: g.add_image(img),
            readback: g.add_buffer(readback),
        };
        // Transfer writes the image (first access -> barrier from UNDEFINED/DST,
        // discard), then Graphics reads it (cross-queue wait + DST->SRC layout
        // transition), copying back to the readback buffer.
        g.add_node::<Upload>(UploadConfig {
            upload,
            region,
        });
        g.add_node::<Readback>(region);

        wait(g.execute(&res, &()));

        {
            let data = buffer_as_slice::<u8>(&readback);
            assert!(data.iter().all(|&b| b == 0x11), "image round-trip mismatch");
        }

        destroy_image(img);
        destroy_buffer(upload);
        destroy_buffer(readback);
    }

    #[test]
    fn compile_once_execute_many() {
        init();
        let buffer = make_buffer();

        let mut g: Graph<Res, ()> = Graph::new();
        let res = Res {
            buf: g.add_buffer(buffer),
        };
        g.add_node::<FillGraphics>(FillConfig {
            value: 0x11,
            size: 64,
        });
        g.compile(&res);

        // Iterations 2.. exercise the hot chain: each frame's fill is ordered
        // against the previous frame's fill (wrap-around WAW).
        for i in 0..8u32 {
            wait(g.execute(&res, &()));
            let data = buffer_as_slice::<u32>(&buffer);
            assert!(data.iter().all(|&x| x == 0x11), "iteration {i} failed");
        }

        destroy_buffer(buffer);
    }

    #[test]
    #[should_panic(expected = "does not belong to this graph")]
    fn foreign_id_in_declare_panics() {
        init();
        let buffer_a = make_buffer();
        let buffer_b = make_buffer();

        // Graph A owns two buffers, so the id we smuggle out has index 1...
        let mut g1: Graph<Res, ()> = Graph::new();
        let _ = g1.add_buffer(buffer_a);
        let foreign = g1.add_buffer(buffer_b);

        // ...which is out of bounds in graph B (one buffer): the writer must
        // catch it at declare time with a clear message.
        struct Bad {
            buf: PassBuffer<TransferWrite>,
        }
        impl Node<Res, ()> for Bad {
            const QUEUE: QueueType = QueueType::Graphics;
            type Config = ();
            fn declare(res: &Res, _: &(), w: &mut NodeWriter) -> Self {
                Bad {
                    buf: w.buffer::<TransferWrite>(res.buf),
                }
            }
            fn execute(&self, _: &(), ctx: &mut ExecutionContext<'_>) {
                let b = ctx.buffer(self.buf);
                ctx.cmd().fill_buffer(&b, 0, 4, 0);
            }
        }

        let mut g2: Graph<Res, ()> = Graph::new();
        let res = Res {
            buf: foreign,
        };
        g2.add_node::<Bad>(());
        let _ = g2.execute(&res, &());
    }
}
