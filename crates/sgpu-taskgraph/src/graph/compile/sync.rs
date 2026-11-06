use sgpu::{AccessType, ImageSubresources, PipelineStage, QueueType};

use crate::{PassImageId, resource::ImageViewRef};

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

#[derive(Default, Debug)]
pub(crate) struct ExecutionBatch {
    pub(crate) nodes: Vec<usize>,
    pub(crate) before_barriers: Barriers,
    pub(crate) first_frame_barriers: Barriers,
}

#[derive(Clone, Debug)]
pub(crate) struct CompiledImageBarrier {
    pub(crate) image: PassImageId,
    pub(crate) view: ImageViewRef,
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
    pub(crate) discard: bool,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct CompiledGlobalBarrier {
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct Barriers {
    pub(crate) images: Vec<CompiledImageBarrier>,
    pub(crate) global: CompiledGlobalBarrier,
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

/// Group Accesses with range opverlapping with provided range, None means whole image
fn access_types(accesses: &[GroupAccess], range: Option<ImageSubresources>) -> Vec<AccessType> {
    let mut types: Vec<AccessType> = Vec::new();
    for access in accesses {
        if ranges_overlap(access.range, range) && !types.contains(&access.access) {
            types.push(access.access);
        }
    }
    types
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

/// The sync for every cell that needs one, in cell space. Splits are decided
/// from this, so a wait and the split it forces can never disagree.
fn derive(timelines: &[Timeline], wave_count: usize) -> Vec<[Option<CellSync>; 3]> {
    let mut cells: Vec<[Option<CellSync>; 3]> = (0..wave_count).map(|_| [None, None, None]).collect();

    for (timeline_index, timeline) in timelines.iter().enumerate() {
        let groups = &timeline.groups;

        // The swapchain must be acquired before the frame's first touch of
        // it, and nothing may be recorded after the present.
        if let Some(present) = groups.last().filter(|last| last.has_present()) {
            let first = groups[0].first_batch();
            cells[first.batch][first.queue].get_or_insert_with(CellSync::default).acquire = true;

            let present_batch = present.first_batch();
            cells[present_batch.batch][present_batch.queue].get_or_insert_with(CellSync::default).terminal = true;
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

            let discard = group == 0 && dst.is_write() && (timeline.transient || src.has_present());

            let stage = dst.stage();

            // Add the wait
            for dst_queue in dst.queues() {
                for src_queue in src.queues().filter(|&queue| queue != dst_queue) {
                    match previous.frame {
                        Frame::Current => add_pending_wait(&mut cells, src.last(src_queue), dst.first(dst_queue), stage),
                        Frame::Previous => {
                            let at = dst.first(dst_queue);
                            push_slot(&mut cells[at.batch][at.queue].get_or_insert_with(CellSync::default).previous_waits, QUEUES[src_queue], stage);
                        }
                    }
                }
            }

            // Add barrier
            match timeline.resource {
                Resource::Image(image) => {
                    let anchor = dst.first_batch();

                    // One barrier per view of `dst` that `src` conflicts with, each
                    // naming only the range it transitions.
                    for (view, range) in conflicting_views(src, dst) {
                        let previous_accesses = access_types(&src.accesses, range);
                        let next = access_types(&dst.accesses, range);
                        let writes = next.iter().any(|access| access.is_write());

                        let steady = CompiledImageBarrier {
                            image,
                            view,
                            previous: previous_accesses.clone(),
                            next: next.clone(),
                            discard: discard && writes,
                        };
                        let first_frame = writes.then(|| CompiledImageBarrier {
                            image,
                            view,
                            previous: vec![AccessType::None],
                            next: next.clone(),
                            discard: true,
                        });

                        let cell = cells[anchor.batch][anchor.queue].get_or_insert_with(CellSync::default);
                        match previous.frame {
                            Frame::Current => {
                                cell.before_barriers.images.push(steady.clone());
                                cell.first_frame_barriers.images.push(steady);
                            }
                            Frame::Previous => {
                                cell.before_barriers.images.push(steady);
                                if let Some(first_frame) = first_frame {
                                    cell.first_frame_barriers.images.push(first_frame);
                                }
                            }
                        }
                    }

                    for dst_queue in dst.queues().filter(|&queue| queue != anchor.queue) {
                        add_pending_wait(&mut cells, anchor, dst.first(dst_queue), stage);
                    }
                }
                Resource::Buffer(_) => {
                    let previous_accesses = access_types(&src.accesses, None);
                    let next = access_types(&dst.accesses, None);

                    for dst_queue in dst.queues() {
                        let dst_batch = dst.first(dst_queue);
                        let cell = cells[dst_batch.batch][dst_batch.queue].get_or_insert_with(CellSync::default);
                        cell.before_barriers.add_global(&previous_accesses, &next);
                        if previous.frame == Frame::Current {
                            cell.first_frame_barriers.add_global(&previous_accesses, &next);
                        }
                    }
                }
            }
        }
    }

    cells
}

/// A wait on `src`, recorded against the cell `dst`; the fold turns it into a
/// submit wait and a signal.
fn add_pending_wait(cells: &mut [[Option<CellSync>; 3]], src: BatchIndex, dst: BatchIndex, stage: PipelineStage) {
    debug_assert!(src < dst, "a wait must point forward: {src:?} -> {dst:?}");
    cells[dst.batch][dst.queue].get_or_insert_with(CellSync::default).waits.push(PendingWait {
        source: src,
        stage,
    });
}

/// The views of `dst` that `src` conflicts with, each with the range it covers
/// (`None` = default view), in access order. A view already covered by an
/// earlier one is skipped: the wider barrier transitions it too.
fn conflicting_views(src: &AccessGroup, dst: &AccessGroup) -> Vec<(ImageViewRef, Option<ImageSubresources>)> {
    let mut views: Vec<(ImageViewRef, Option<ImageSubresources>)> = Vec::new();

    for access in &dst.accesses {
        let needed = src.accesses.iter().any(|previous| ranges_overlap(previous.range, access.range) && access_conflicts(previous.access, access.access));
        if needed && !views.iter().any(|(view, range)| *view == access.view || range_contains(*range, access.range)) {
            views.push((access.view, access.range));
        }
    }

    views
}

/// Folds the schedule's waves into submits
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
            let exec = ExecutionBatch {
                nodes: members,
                ..Default::default()
            };

            let (index, batch) = match open[queue] {
                Some(index) if !split => {
                    let batch = submits[queue][index].exec_batches.len();
                    submits[queue][index].exec_batches.push(exec);
                    (index, batch)
                }
                _ => {
                    let index = push_submit(
                        &mut submits[queue],
                        SubmitBatch {
                            queue: QUEUES[queue],
                            exec_batches: vec![exec],
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

    let exec = &mut submits[queue][index].exec_batches[batch];
    exec.before_barriers = before_barriers;
    exec.first_frame_barriers = first_frame_barriers;

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

        match submits[queue][index].waits.iter_mut().find(|wait| wait.source == source) {
            Some(existing) => {
                existing.stage |= wait.stage;
            }
            None => {
                submits[queue][index].waits.push(Wait {
                    source,
                    stage: wait.stage,
                });

                submits[source.queue][source.index].signals.push(target);
            }
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
