use sgpu::{AccessType, AcquiredImage, Counter, GlobalBarrier, ImageBarrier, ImageLayout, PipelineStage};
use smallvec::SmallVec;

use super::TaskGraph;
use crate::node::ExecutionContext;

use super::ErasedNode;
use crate::graph::compile::sync::{CompiledImageBarrier, CompiledSync};

/// One frame's submission state: the plan is interpreted cell by cell, and
/// a cell records exactly once — on demand, into a command buffer of its
/// own.
struct Frame {
    /// Counters of submitted cells, indexed (queue, cell); `None` marks a
    /// pending cell. Seeded empty: a cell's counter exists once it flushed.
    counters: [Vec<Option<Counter>>; 3],
    /// Next unrecorded cell index per queue.
    head: [usize; 3],
    /// Last submitted counter per queue — the book write-back source.
    last: [Option<Counter>; 3],
    /// Any last submission — a headless frame's completion counter.
    last_any: Option<Counter>,
    /// The terminal present cell's counter.
    present_counter: Option<Counter>,
}

impl Frame {
    fn new(cells: &CompiledSync) -> Frame {
        Frame {
            counters: std::array::from_fn(|q| vec![None; cells.cells[q].len()]),
            head: [0; 3],
            last: [None; 3],
            last_any: None,
            present_counter: None,
        }
    }
}

/// Everything one cell record needs; the flags are frame-constants copied
/// before the run loop so the runtime books never borrow across it.
struct Checkpoints<'a, 'g, R: 'g, D: 'g> {
    graph: &'a TaskGraph<'g, R, D>,
    built: &'a [Box<dyn ErasedNode<R, D> + 'g>],
    cells: &'a CompiledSync,
    acquired: Option<&'a AcquiredImage>,
    data: &'a D,
    /// This frame's slot.
    curr: usize,
    /// Books carry a full plan cycle: steady barrier lists stroke, first
    /// cycles claim no previous state (undefined lists instead).
    steady: bool,
    cyclic_fresh: bool,
    /// This slot's ring — the previous cycle's last counters per queue.
    ring: [Option<Counter>; 3],
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    pub(super) fn run(&mut self, acquired: Option<&AcquiredImage>, data: &D) -> Counter {
        // The compiled graph is taken out for the duration of the run:
        // shape references (cells/built) and the runtime-book updates
        // borrow it disjointly, and the end of this function puts it back
        // for the next frame. A panic in between drops the graph — the
        // next call correctly refuses with "not compiled".
        let mut compiled = self.compiled.take().expect("TaskGraph is not compiled: call compile(res) once after building it, and again after every mutation or set_frames_in_flight() (resize handles its own refresh)");

        // Resolve this frame's slot. The swapchain's slot comes from the
        // acquire; headless graphs advance their own ring. Swapchain slots
        // are fence-paced by the swapchain itself, so slot `k` is only
        // reused after slot `k`'s previous frame completed on the GPU. The
        // app rebinds the acquired image into its external rack itself
        // (`set_image` before executing).
        if let Some(acquired) = acquired {
            assert!(acquired.frame_index() < self.frames_in_flight, "acquired slot {} out of range for slot_count {}; set_frames_in_flight must match Swapchain::frame_count", acquired.frame_index(), self.frames_in_flight);
            compiled.runtime.curr_frame = acquired.frame_index();
        } else if self.frames_in_flight == 0 {
            panic!("TaskGraph::execute requires set_frames_in_flight(F) first (use execute_with_swapchain for swapchain graphs)");
        } else {
            compiled.runtime.curr_frame = (compiled.runtime.curr_frame + 1) % self.frames_in_flight;
        }

        let curr = compiled.runtime.curr_frame;
        let steady = compiled.runtime.executed_once;
        let cyclic_fresh = !compiled.runtime.frame_executed[curr];
        // The rolling counters seed from the previous frame's rotation
        // position: its slot is (curr - 1 + F) % F. Both modes rotate — a
        // headless graph advances its own ring, and the swapchain's fence
        // pacing turns every execute into exactly the next position.
        let seeded = compiled.runtime.per_frame_counters[(curr + self.frames_in_flight - 1) % self.frames_in_flight];
        let ring = compiled.runtime.per_frame_counters[curr];

        // Create or recreate the image views of currently bound images
        self.views.refresh(&self.images);

        let mut frame = Frame::new(&compiled.sync);

        {
            let checkpoints = Checkpoints {
                graph: self,
                built: &compiled.built,
                cells: &compiled.sync,
                acquired,
                data,
                curr,
                steady,
                cyclic_fresh,
                ring,
            };

            // The plan is the record order: visit cells in ascending (wave,
            // queue); the visit itself is a flush marker. Anything referenced
            // earlier flushes on demand inside `flush_until`.
            for &(q, k) in &compiled.sync.order {
                flush_until(&checkpoints, &mut frame, q as usize, k);
            }
            // Drain the remainder of every queue's plan column.
            for (q, _) in checkpoints.cells.cells.iter().enumerate() {
                if let Some(last) = checkpoints.cells.cells[q].len().checked_sub(1) {
                    flush_until(&checkpoints, &mut frame, q, last);
                }
            }
        }

        // Books: per queue, the last counter this frame flushed, or the
        // seed carried over for queues that never submitted (both modes).
        for (qi_id, _) in compiled.sync.cells.iter().enumerate() {
            compiled.runtime.per_frame_counters[curr][qi_id] = frame.last[qi_id].or(seeded[qi_id]);
        }
        compiled.runtime.frame_executed[curr] = true;
        compiled.runtime.executed_once = true;

        self.compiled = Some(compiled);
        frame.present_counter.or(frame.last_any).expect("graph has no nodes")
    }
}

/// Submits every unrecorded cell of queue `q` up to and including `target`,
/// in the queue's own stream order. Exact wait addressing is exactly this:
/// a receiver only forces the referenced source cell out, later cells of
/// the source queue stay pending and overlap.
fn flush_until<'g, R: 'g, D: 'g>(checkpoints: &Checkpoints<'_, 'g, R, D>, frame: &mut Frame, q: usize, target: usize) {
    debug_assert!(q < 3, "queue index {q} out of the registry");
    debug_assert!(target < checkpoints.cells.cells[q].len(), "cell {target} referenced out of queue {q}'s plan");
    while frame.head[q] <= target {
        record_cell(checkpoints, frame, q, frame.head[q]);
        frame.head[q] += 1;
    }
}

/// Records and submits one cell: resolves its waits (which recurses into
/// flush_until across queues), records barriers and nodes into a fresh
/// command buffer, and submits — one cell, one submission, one counter.
fn record_cell<'g, R: 'g, D: 'g>(checkpoints: &Checkpoints<'_, 'g, R, D>, frame: &mut Frame, q: usize, k: usize) {
    let cell = &checkpoints.cells.cells[q][k];

    let mut waits: SmallVec<[(Counter, PipelineStage); 3]> = SmallVec::new();
    // Exact same-frame edges: force the referenced source cell out, then
    // wait its submission counter.
    for &(src_q, src_k, stage) in &cell.waits {
        let src = src_q as usize;
        flush_until(checkpoints, frame, src, src_k);
        let counter = frame.counters[src][src_k].expect("exact wait source not submitted");
        waits.push((counter, stage));
    }
    // Cross-cycle edges: the same slot's previous cycle produced the ring's
    // counters last frame — nothing to flush, just wait them.
    for &(src, stage) in &cell.slot_waits {
        if let Some(counter) = checkpoints.ring[src as usize] {
            waits.push((counter, stage));
        }
    }

    let mut cmd = sgpu::record(cell.queue);

    if cell.acquire
        && let Some(acquired) = checkpoints.acquired
    {
        cmd.wait_for_swapchain_image(acquired);
    }
    for (counter, stage) in waits {
        cmd.wait_for(counter, stage);
    }

    let barriers = if checkpoints.steady {
        &cell.barriers
    } else {
        &cell.undefined_barriers
    };
    if !barriers.images.is_empty() || !barriers.global.previous.is_empty() {
        let global = (!barriers.global.previous.is_empty()).then(|| GlobalBarrier {
            previous_accesses: &barriers.global.previous,
            next_accesses: &barriers.global.next,
        });

        let mut imgs: SmallVec<[ImageBarrier; 8]> = SmallVec::new();
        for barrier in &barriers.images {
            // Cyclic racks (rings, transient heap members) on a slot's
            // first-ever cycle claim no previous state: substitute the
            // first-use discard variant (the undefined list never contains
            // the layout claim for this image).
            let barrier: &CompiledImageBarrier = if barrier.is_graph_owned && checkpoints.cyclic_fresh {
                cell.undefined_barriers.images.iter().find(|candidate| candidate.image == barrier.image && candidate.previous == [AccessType::None]).expect("cyclic barrier must have an undefined first-use variant")
            } else {
                barrier
            };
            imgs.push(ImageBarrier {
                view: checkpoints.graph.images[barrier.image as usize].image_at(checkpoints.curr).default_view(),
                previous_accesses: &barrier.previous,
                next_accesses: &barrier.next,
                previous_layout: ImageLayout::Optimal,
                next_layout: ImageLayout::Optimal,
                discard_contents: barrier.discard,
            });
        }
        cmd.barriers(global.as_ref(), &imgs);
    }

    {
        let mut ctx = ExecutionContext {
            cmd: &mut cmd,
            images: &checkpoints.graph.images,
            buffers: &checkpoints.graph.buffers,
            views: &checkpoints.graph.views,
            curr_frame: checkpoints.curr,
        };
        for &n in &cell.nodes {
            checkpoints.built[n].execute(checkpoints.data, &mut ctx);
        }
    }

    let counter = sgpu::submit(&[cmd]);
    frame.counters[q][k] = Some(counter);
    frame.last[q] = Some(counter);
    frame.last_any = Some(counter);
    if cell.terminal {
        frame.present_counter = Some(counter);
    }
}
