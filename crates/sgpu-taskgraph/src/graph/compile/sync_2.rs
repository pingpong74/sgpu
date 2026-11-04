use sgpu::{AccessType, PipelineStage, QueueType};

use crate::PassImageId;

use super::{NodeDeclaration, QUEUES, schedule::Schedule, timeline::*};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct SubmitId {
    pub(crate) queue: usize,
    pub(crate) index: usize,
}

#[derive(Debug)]
pub(crate) struct SubmitBatch {
    pub(crate) queue: QueueType,
    pub(crate) exec_batches: Vec<ExecutionBatch>,
    pub(crate) waits: Vec<Wait>,
    pub(crate) previous_waits: Vec<SlotWait>,
    pub(crate) signals: Vec<SubmitId>,
    pub(crate) acquire: bool,
    pub(crate) terminal: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Wait {
    pub(crate) source: SubmitId,
    pub(crate) stage: PipelineStage,
}

/// Waits on the last counter from a given queue from last frame
#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotWait {
    pub(crate) queue: QueueType,
    pub(crate) stage: PipelineStage,
}

/// Nodes to be recorded in one cmd buffer
/// first frame barriers are to be used for the first Batch
#[derive(Debug)]
pub(crate) struct ExecutionBatch {
    pub(crate) nodes: Vec<usize>,
    pub(crate) before_barriers: Barriers,
    pub(crate) first_frame_barriers: Barriers,
}

/// Image layout/state transitions plus a memory barrier, as recorded.
#[derive(Clone, Default, Debug)]
pub(crate) struct Barriers {
    pub(crate) images: Vec<CompiledImageBarrier>,
    pub(crate) global: CompiledGlobalBarrier,
}

#[derive(Clone, Debug)]
pub(crate) struct CompiledImageBarrier {
    pub(crate) image: PassImageId,
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
    pub(crate) discard: bool,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct CompiledGlobalBarrier {
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
}

impl Barriers {
    pub(crate) fn add_global(&mut self, previous: &[AccessType], next: &[AccessType]) {
        for &access in previous {
            if !self.global.previous.contains(&access) {
                self.global.previous.push(access);
            }
        }
        for &access in next {
            if !self.global.next.contains(&access) {
                self.global.next.push(access);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct SubmitPlan {
    pub(crate) submits: [Vec<SubmitBatch>; 3],
}

/// Everything that must be applied to one schedule cell before its nodes run.
///
/// Waits and acquire are submit-level, so a cell carrying them has to start
/// its own submit; barriers and slot waits travel with the cell and do not
/// split.
#[derive(Default)]
struct CellSync {
    waits: Vec<PendingWait>,
    previous_waits: Vec<SlotWait>,
    before_barriers: Barriers,
    first_frame_barriers: Barriers,
    acquire: bool,
    terminal: bool,
}

impl CellSync {
    fn splits(&self) -> bool { !self.waits.is_empty() || self.acquire || self.terminal }
}

/// A wait whose source is still a cell; the fold resolves it to a submit.
struct PendingWait {
    source: BatchIndex,
    stage: PipelineStage,
}

/// Records a wait, OR-ing the stage into an existing one for the same source
/// so a submit makes at most one `wait_for` per predecessor submit. Returns
/// whether a new wait was recorded.
fn push_wait(waits: &mut Vec<Wait>, source: SubmitId, stage: PipelineStage) -> bool {
    match waits.iter_mut().find(|wait| wait.source == source) {
        Some(existing) => {
            existing.stage |= stage;
            false
        }
        None => {
            waits.push(Wait {
                source,
                stage,
            });
            true
        }
    }
}

/// As `push_wait`, but a slot wait's source is a whole queue's previous-frame
/// counter, so at most one per queue.
fn push_slot(slots: &mut Vec<SlotWait>, queue: QueueType, stage: PipelineStage) {
    match slots.iter_mut().find(|slot| slot.queue == queue) {
        Some(existing) => existing.stage |= stage,
        None => slots.push(SlotWait {
            queue,
            stage,
        }),
    }
}

/// Appends `submit` and links the previous submit on its queue to it, so
/// queue order survives a drain that picks ready submits in any order.
/// Returns the new submit's index.
fn push_submit(column: &mut Vec<SubmitBatch>, submit: SubmitBatch) -> usize {
    let index = column.len();
    if index > 0 {
        column[index - 1].signals.push(SubmitId {
            queue: submit.queue as usize,
            index,
        });
    }
    column.push(submit);
    index
}

/// The cell's sync, created on first use.
fn cell_mut(cells: &mut [[Option<CellSync>; 3]], at: BatchIndex) -> &mut CellSync { cells[at.batch][at.queue].get_or_insert_with(CellSync::default) }

/// The sync for every cell that needs one, in cell space. Splits are decided
/// from this, so a wait and the split it forces can never disagree.
fn derive(timelines: &[Timeline], wave_count: usize) -> Vec<[Option<CellSync>; 3]> {
    let mut cells: Vec<[Option<CellSync>; 3]> = (0..wave_count).map(|_| [None, None, None]).collect();

    for (timeline_index, timeline) in timelines.iter().enumerate() {
        let groups = &timeline.groups;

        // The swapchain must be acquired before the frame's first touch of
        // it, and nothing may be recorded after the present.
        if groups.last().is_some_and(AccessGroup::is_present) {
            let first = groups[0].first_batch();
            cell_mut(&mut cells, first).acquire = true;

            let present = groups[groups.len() - 1].first_batch();
            cell_mut(&mut cells, present).terminal = true;
        }

        for (group, dst) in groups.iter().enumerate() {
            let previous = if group == 0 {
                timeline.previous_use
            } else {
                PreviousUse {
                    timeline: timeline_index,
                    group: group - 1,
                    frame: Frame::Current,
                }
            };
            let src = &timelines[previous.timeline].groups[previous.group];

            if previous.timeline == timeline_index && !conflicts(&src.accesses, &dst.accesses) {
                continue;
            }

            let transient = timeline.transient;
            // Do we need the is_write condition tho
            let discard = group == 0 && is_write(&dst.accesses) && (transient || src.accesses.contains(&AccessType::Present));

            add_wait(&mut cells, src, dst, previous.frame);
            add_barrier(&mut cells, src, dst, timeline.resource, previous.frame, discard);
        }
    }

    cells
}

/// Waits so `dst` starts after `src` finished on every queue but the one
/// `dst` already runs on, where submission order does it.
fn add_wait(cells: &mut [[Option<CellSync>; 3]], src: &AccessGroup, dst: &AccessGroup, frame: Frame) {
    let stage = stage_of(&dst.accesses);

    for dst_queue in dst.queues() {
        for src_queue in src.queues().filter(|&queue| queue != dst_queue) {
            match frame {
                Frame::Current => add_pending_wait(cells, src.last(src_queue), dst.first(dst_queue), stage),
                Frame::Previous => {
                    let at = dst.first(dst_queue);
                    push_slot(&mut cell_mut(cells, at).previous_waits, QUEUES[src_queue], stage);
                }
            }
        }
    }
}

/// A wait on `src`, recorded against the cell `dst`; the fold turns it into a
/// submit wait and a signal.
fn add_pending_wait(cells: &mut [[Option<CellSync>; 3]], src: BatchIndex, dst: BatchIndex, stage: PipelineStage) {
    debug_assert!(src < dst, "a wait must point forward: {src:?} -> {dst:?}");
    cell_mut(cells, dst).waits.push(PendingWait {
        source: src,
        stage,
    });
}

/// The barriers that make `dst` safe after `src`.
fn add_barrier(cells: &mut [[Option<CellSync>; 3]], src: &AccessGroup, dst: &AccessGroup, resource: Resource, frame: Frame, discard: bool) {
    let stage = stage_of(&dst.accesses);

    match resource {
        Resource::Image(image) => {
            // The layout transition runs on the group's anchor queue;
            // the other queues that touch it wait for that submit.
            let anchor = dst.first_batch();

            let steady = CompiledImageBarrier {
                image,
                previous: src.accesses.clone(),
                next: dst.accesses.clone(),
                discard,
            };
            // On a slot that never ran there is no previous state to
            // order against: transition from nothing instead, and drop
            // whatever the image was created with.
            let first_frame = is_write(&dst.accesses).then(|| CompiledImageBarrier {
                image,
                previous: vec![AccessType::None],
                next: dst.accesses.clone(),
                discard: true,
            });

            let cell = cell_mut(cells, anchor);
            match frame {
                // `src` is earlier in this frame, so it ran on the first
                // frame too and the same barrier serves both.
                Frame::Current => {
                    cell.before_barriers.images.push(steady.clone());
                    cell.first_frame_barriers.images.push(steady);
                }
                // `src` is last frame's: only steady can name it. First
                // frame takes the from-nothing transition, if any.
                Frame::Previous => {
                    cell.before_barriers.images.push(steady);
                    if let Some(first_frame) = first_frame {
                        cell.first_frame_barriers.images.push(first_frame);
                    }
                }
            }

            for dst_queue in dst.queues().filter(|&queue| queue != anchor.queue) {
                add_pending_wait(cells, anchor, dst.first(dst_queue), stage);
            }
        }
        Resource::Buffer(_) => {
            // Buffers have no layout, so a memory barrier on every queue
            // that touches them.
            for dst_queue in dst.queues() {
                let cell = cell_mut(cells, dst.first(dst_queue));
                cell.before_barriers.add_global(&src.accesses, &dst.accesses);
                if frame == Frame::Current {
                    cell.first_frame_barriers.add_global(&src.accesses, &dst.accesses);
                }
            }
        }
    }
}

/// Folds the schedule's waves into submits — one column per queue — and
/// attaches the sync `derive` placed on each cell.
///
/// Cells are visited in `BatchIndex` order, so when a wait is attached its
/// source submit has already been folded and can be resolved directly.
fn build_submits<'g, R, D>(schedule: &Schedule, nodes: &[NodeDeclaration<'g, R, D>], mut cells: Vec<[Option<CellSync>; 3]>) -> [Vec<SubmitBatch>; 3] {
    let mut submits: [Vec<SubmitBatch>; 3] = std::array::from_fn(|_| Vec::new());
    let mut open: [Option<usize>; 3] = [None; 3];
    let mut submit_of: Vec<[Option<usize>; 3]> = (0..schedule.batches.len()).map(|_| [None; 3]).collect();

    for (wave, wave_nodes) in schedule.batches.iter().enumerate() {
        for queue in 0..3 {
            let members: Vec<usize> = wave_nodes.iter().copied().filter(|&node| nodes[node].queue as usize == queue).collect();
            if members.is_empty() {
                continue;
            }

            let sync = cells[wave][queue].take();
            let split = sync.as_ref().is_some_and(CellSync::splits);

            let (index, batch) = match open[queue] {
                Some(index) if !split => {
                    let batch = submits[queue][index].exec_batches.len();
                    submits[queue][index].exec_batches.push(ExecutionBatch {
                        nodes: members,
                        before_barriers: Barriers::default(),
                        first_frame_barriers: Barriers::default(),
                    });
                    (index, batch)
                }
                _ => {
                    let index = push_submit(
                        &mut submits[queue],
                        SubmitBatch {
                            queue: QUEUES[queue],
                            exec_batches: vec![ExecutionBatch {
                                nodes: members,
                                before_barriers: Barriers::default(),
                                first_frame_barriers: Barriers::default(),
                            }],
                            waits: Vec::new(),
                            previous_waits: Vec::new(),
                            signals: Vec::new(),
                            acquire: false,
                            terminal: false,
                        },
                    );
                    open[queue] = Some(index);
                    (index, 0)
                }
            };
            submit_of[wave][queue] = Some(index);

            if let Some(sync) = sync {
                let terminal = sync.terminal;
                attach_sync(&mut submits, &submit_of, queue, index, batch, sync);
                if terminal {
                    open[queue] = None;
                }
            }
        }
    }

    debug_assert!(cells.iter().flatten().all(|cell| cell.is_none()), "every cell sync was attached");

    submits
}

/// Attaches one cell's sync to the submit and batch that carry it.
fn attach_sync(submits: &mut [Vec<SubmitBatch>; 3], submit_of: &[[Option<usize>; 3]], queue: usize, index: usize, batch: usize, sync: CellSync) {
    let CellSync {
        waits,
        previous_waits,
        before_barriers,
        first_frame_barriers,
        acquire,
        terminal,
    } = sync;

    debug_assert!(batch == 0 || (waits.is_empty() && !acquire && !terminal), "waits, acquire and present must start their submit");

    {
        let exec = &mut submits[queue][index].exec_batches[batch];
        exec.before_barriers = before_barriers;
        exec.first_frame_barriers = first_frame_barriers;
    }

    submits[queue][index].acquire |= acquire;
    submits[queue][index].terminal |= terminal;
    for slot in previous_waits {
        push_slot(&mut submits[queue][index].previous_waits, slot.queue, slot.stage);
    }

    let target = SubmitId {
        queue,
        index,
    };
    for wait in waits {
        let source = SubmitId {
            queue: wait.source.queue,
            index: submit_of[wait.source.batch][wait.source.queue].expect("a wait source is folded before its destination"),
        };
        let recorded = push_wait(&mut submits[queue][index].waits, source, wait.stage);
        if recorded {
            submits[source.queue][source.index].signals.push(target);
        }
    }
}

/// Plans every submission of a scheduled, alias-resolved graph: the sync from
/// `derive`, folded into submits by `build_submits`.
pub(crate) fn build_submit_plan<'g, R, D>(timelines: &[Timeline], schedule: &Schedule, nodes: &[NodeDeclaration<'g, R, D>]) -> SubmitPlan {
    let cells = derive(timelines, schedule.batches.len());

    SubmitPlan {
        submits: build_submits(schedule, nodes, cells),
    }
}
