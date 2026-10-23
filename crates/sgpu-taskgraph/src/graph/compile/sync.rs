use sgpu::{AccessType, PipelineStage, QueueType};

use super::{
    QUEUES,
    alias::{AliasPlan, Frame, PreviousUse},
    schedule::{GroupSpan, Schedule, SliceId},
    timeline::{Resource, Timeline, conflicts, is_write, stage_of},
};
use crate::TaskGraph;

#[derive(Clone, Debug)]
pub(crate) struct CompiledImageBarrier {
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
    pub(crate) discard: bool,
    pub(crate) image: u32,
}

#[derive(Default, Debug)]
pub(crate) struct CompiledGlobalBarrier {
    pub(crate) previous: Vec<AccessType>,
    pub(crate) next: Vec<AccessType>,
}

// Barriers used on the graph's first frame.
#[derive(Default, Debug)]
pub(crate) struct Barriers {
    pub(crate) images: Vec<CompiledImageBarrier>,
    pub(crate) global: CompiledGlobalBarrier,
}

impl Barriers {
    fn is_empty(&self) -> bool { self.images.is_empty() && self.global.previous.is_empty() && self.global.next.is_empty() }

    fn add_global(&mut self, previous: &[AccessType], next: &[AccessType]) {
        extend_unique(&mut self.global.previous, previous);
        extend_unique(&mut self.global.next, next);
    }

    fn absorb(&mut self, other: Barriers) {
        self.images.extend(other.images);
        self.add_global(&other.global.previous, &other.global.next);
    }
}

#[derive(Debug)]
pub(crate) struct SteadyImageBarrier {
    pub(crate) barrier: CompiledImageBarrier,
    // Used instead of `barrier` when this slot has never run.
    pub(crate) first_use: Option<CompiledImageBarrier>,
}

// Barriers used after the graph's first frame.
#[derive(Default, Debug)]
pub(crate) struct SteadyBarriers {
    pub(crate) images: Vec<SteadyImageBarrier>,
    pub(crate) global: CompiledGlobalBarrier,
}

impl SteadyBarriers {
    fn is_empty(&self) -> bool { self.images.is_empty() && self.global.previous.is_empty() && self.global.next.is_empty() }

    fn add_global(&mut self, previous: &[AccessType], next: &[AccessType]) {
        extend_unique(&mut self.global.previous, previous);
        extend_unique(&mut self.global.next, next);
    }

    fn absorb(&mut self, other: SteadyBarriers) {
        self.images.extend(other.images);
        self.add_global(&other.global.previous, &other.global.next);
    }
}

#[derive(Default, Debug)]
pub(crate) struct BarrierSet {
    pub(crate) first_frame: Barriers,
    pub(crate) steady: SteadyBarriers,
}

impl BarrierSet {
    fn is_empty(&self) -> bool { self.first_frame.is_empty() && self.steady.is_empty() }

    fn add_current(&mut self, barrier: CompiledImageBarrier) {
        self.first_frame.images.push(barrier.clone());
        self.steady.images.push(SteadyImageBarrier {
            barrier,
            first_use: None,
        });
    }

    fn add_previous(&mut self, barrier: CompiledImageBarrier) {
        self.steady.images.push(SteadyImageBarrier {
            barrier,
            first_use: None,
        });
    }

    fn add_previous_with_first_use(&mut self, barrier: CompiledImageBarrier, first_use: CompiledImageBarrier) {
        self.first_frame.images.push(first_use.clone());
        self.steady.images.push(SteadyImageBarrier {
            barrier,
            first_use: Some(first_use),
        });
    }

    fn add_first_frame(&mut self, barrier: CompiledImageBarrier) { self.first_frame.images.push(barrier); }

    fn add_global_current(&mut self, previous: &[AccessType], next: &[AccessType]) {
        self.first_frame.add_global(previous, next);
        self.steady.add_global(previous, next);
    }

    fn add_global_previous(&mut self, previous: &[AccessType], next: &[AccessType]) { self.steady.add_global(previous, next); }

    fn absorb(&mut self, other: BarrierSet) {
        self.first_frame.absorb(other.first_frame);
        self.steady.absorb(other.steady);
    }
}

fn extend_unique(into: &mut Vec<AccessType>, from: &[AccessType]) {
    for &access in from {
        if !into.contains(&access) {
            into.push(access);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Wait {
    pub(crate) queue: QueueType,
    // Source wave while building, source submission after `resolve_waits`.
    pub(crate) source: usize,
    pub(crate) stage: PipelineStage,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SlotWait {
    pub(crate) queue: QueueType,
    pub(crate) stage: PipelineStage,
}

// One submission, or a candidate that may merge into one.
#[derive(Debug)]
pub(crate) struct Submission {
    pub(crate) queue: QueueType,
    pub(crate) wave: usize,
    pub(crate) nodes: Vec<usize>,
    pub(crate) waits: Vec<Wait>,
    pub(crate) slot_waits: Vec<SlotWait>,
    pub(crate) acquire: bool,
    pub(crate) terminal: bool,
    pub(crate) barriers: BarrierSet,
    // A wait references this candidate, so its submission must end here.
    is_wait_target: bool,
}

impl Submission {
    fn new(queue: QueueType, wave: usize, nodes: Vec<usize>, terminal: bool) -> Self {
        Self {
            queue,
            wave,
            nodes,
            waits: Vec::new(),
            slot_waits: Vec::new(),
            acquire: false,
            terminal,
            barriers: BarrierSet::default(),
            is_wait_target: false,
        }
    }

    fn is_plain(&self) -> bool { self.waits.is_empty() && self.slot_waits.is_empty() && !self.acquire && self.barriers.is_empty() }

    fn absorb(&mut self, other: Submission) {
        debug_assert_eq!(self.queue, other.queue);
        self.nodes.extend(other.nodes);
        for wait in other.waits {
            push_exact(&mut self.waits, wait.queue, wait.source, wait.stage);
        }
        for slot in other.slot_waits {
            push_slot(&mut self.slot_waits, slot.queue, slot.stage);
        }
        self.acquire |= other.acquire;
        self.terminal |= other.terminal;
        self.barriers.absorb(other.barriers);
    }
}

// Per-queue submissions and the order in which the executor records them.
#[derive(Debug)]
pub(crate) struct SubmissionPlan {
    pub(crate) submissions: [Vec<Submission>; 3],
    pub(crate) order: Vec<(QueueType, usize)>,
}

fn qi(queue: QueueType) -> usize { QUEUES.iter().position(|&q| q == queue).expect("node queue is a graph queue") }

#[derive(Clone, Copy)]
struct AccessPoint<'a> {
    span: GroupSpan,
    accesses: &'a [AccessType],
    stage: PipelineStage,
}

impl<'g, R: 'g, D: 'g> TaskGraph<'g, R, D> {
    pub(crate) fn build_submission_plan(&self, timelines: &[Timeline], schedule: &Schedule, alias: &AliasPlan) -> SubmissionPlan {
        let owned: Vec<bool> = timelines
            .iter()
            .map(|tl| match tl.resource {
                Resource::Image(i) => self.images[i as usize].is_graph_owned(),
                Resource::Buffer(i) => self.buffers[i as usize].is_graph_owned(),
            })
            .collect();
        let transient: Vec<bool> = timelines
            .iter()
            .map(|tl| match tl.resource {
                Resource::Image(i) => self.images[i as usize].is_transient(),
                Resource::Buffer(i) => self.buffers[i as usize].is_transient(),
            })
            .collect();

        let mut present = vec![false; schedule.node_count()];
        for timeline in timelines {
            if let Some(group) = timeline.groups.last().filter(|group| group.is_present()) {
                present[group.members[0]] = true;
            }
        }

        let mut raw: Vec<[Option<Submission>; 3]> = (0..schedule.batches.len()).map(|_| [None, None, None]).collect();
        for (wave, batch) in schedule.batches.iter().enumerate() {
            for queue in 0..3 {
                let nodes: Vec<usize> = batch.iter().copied().filter(|&node| schedule.queue_of(node) == queue).collect();
                if nodes.is_empty() {
                    continue;
                }
                let terminal = nodes.iter().any(|&node| present[node]);
                raw[wave][queue] = Some(Submission::new(QUEUES[queue], wave, nodes, terminal));
            }
        }

        PlanBuilder {
            timelines,
            schedule,
            alias,
            owned,
            transient,
            raw,
        }
        .build()
    }
}

struct PlanBuilder<'a> {
    timelines: &'a [Timeline],
    schedule: &'a Schedule,
    alias: &'a AliasPlan,
    owned: Vec<bool>,
    transient: Vec<bool>,
    /// `[batch][queue]`
    raw: Vec<[Option<Submission>; 3]>,
}

impl<'a> PlanBuilder<'a> {
    fn build(mut self) -> SubmissionPlan {
        self.derive();
        self.finish()
    }

    fn side(&self, timeline: usize, group: usize) -> AccessPoint<'a> {
        let accesses = &self.timelines[timeline].groups[group].accesses;
        AccessPoint {
            span: self.schedule.span(timeline, group),
            accesses,
            stage: stage_of(accesses),
        }
    }

    fn raw_mut(&mut self, id: SliceId) -> &mut Submission { self.raw[id.wave][id.queue].as_mut().expect("sync targets a slice with no nodes") }

    fn derive(&mut self) {
        for timeline in 0..self.timelines.len() {
            let group_count = self.timelines[timeline].groups.len();
            for group in 0..group_count {
                let dst = self.side(timeline, group);
                if group == 0 && self.timelines[timeline].groups[group_count - 1].is_present() {
                    self.raw_mut(dst.span.anchor()).acquire = true;
                }

                let previous = if group == 0 {
                    self.alias.previous_use[timeline]
                } else {
                    PreviousUse {
                        timeline,
                        group: group - 1,
                        frame: Frame::Current,
                    }
                };
                let src = self.side(previous.timeline, previous.group);

                // A timeline that follows itself only needs sync on a real conflict.
                if previous.timeline == timeline && !conflicts(src.accesses, dst.accesses) {
                    continue;
                }

                let discard = group == 0 && is_write(dst.accesses) && (self.transient[timeline] || src.accesses.contains(&AccessType::Present));
                let resource = self.timelines[timeline].resource;
                let owned = self.owned[timeline];

                // Cross-frame edges only order graph-owned resources. External
                // resources are not ours to wait on across slots.
                if previous.frame == Frame::Current || owned {
                    self.add_wait(src, dst, previous.frame);
                }
                self.add_barrier(src, dst, resource, previous.frame, discard, owned);
            }
        }
    }

    fn add_explicit_wait(&mut self, src: SliceId, dst: SliceId, stage: PipelineStage) {
        debug_assert!(src < dst, "a wait must point backwards in submission order: {src:?} -> {dst:?}");
        self.raw_mut(src).is_wait_target = true;
        let dst = self.raw_mut(dst);
        push_exact(&mut dst.waits, QUEUES[src.queue], src.wave, stage);
    }

    // Waits so `dst` starts after `src` finishes on every other queue.
    fn add_wait(&mut self, src: AccessPoint<'a>, dst: AccessPoint<'a>, frame: Frame) {
        for dst_queue in dst.span.queues() {
            for src_queue in src.span.queues().filter(|&queue| queue != dst_queue) {
                match frame {
                    Frame::Current => {
                        let src_id = src.span.last(src_queue);
                        let dst_id = dst.span.first(dst_queue);
                        self.add_explicit_wait(src_id, dst_id, dst.stage);
                    }
                    Frame::Previous => {
                        let dst_id = dst.span.first(dst_queue);
                        let events = self.raw_mut(dst_id);
                        push_slot(&mut events.slot_waits, QUEUES[src_queue], dst.stage);
                    }
                }
            }
        }
    }

    // The barrier that makes `dst` safe after `src`.
    fn add_barrier(&mut self, src: AccessPoint<'a>, dst: AccessPoint<'a>, resource: Resource, frame: Frame, discard: bool, owned: bool) {
        match resource {
            Resource::Image(image) => {
                let host = dst.span.anchor();
                let barrier = CompiledImageBarrier {
                    image,
                    previous: src.accesses.to_vec(),
                    next: dst.accesses.to_vec(),
                    discard,
                };
                let first_use = (frame == Frame::Previous && is_write(dst.accesses)).then(|| CompiledImageBarrier {
                    image,
                    previous: vec![AccessType::None],
                    next: dst.accesses.to_vec(),
                    discard: true,
                });

                {
                    let events = self.raw_mut(host);
                    match frame {
                        Frame::Current => events.barriers.add_current(barrier),
                        Frame::Previous if owned => match first_use {
                            Some(first_use) => events.barriers.add_previous_with_first_use(barrier, first_use),
                            None => events.barriers.add_previous(barrier),
                        },
                        Frame::Previous => {
                            events.barriers.add_previous(barrier);
                            if let Some(first_use) = first_use {
                                events.barriers.add_first_frame(first_use);
                            }
                        }
                    }
                }

                // The transition runs on the host queue; other queues wait for it.
                for dst_queue in dst.span.queues().filter(|&queue| queue != host.queue) {
                    let dst_id = dst.span.first(dst_queue);
                    self.add_explicit_wait(host, dst_id, dst.stage);
                }
            }
            Resource::Buffer(_) => {
                for dst_queue in dst.span.queues() {
                    let dst_id = dst.span.first(dst_queue);
                    let events = self.raw_mut(dst_id);
                    match frame {
                        Frame::Current => events.barriers.add_global_current(src.accesses, dst.accesses),
                        Frame::Previous => events.barriers.add_global_previous(src.accesses, dst.accesses),
                    }
                }
            }
        }
    }

    fn finish(self) -> SubmissionPlan {
        let (mut submissions, submission_of_wave) = compact(self.raw, self.schedule.batches.len());
        resolve_waits(&mut submissions, &submission_of_wave);

        let mut order = submissions.iter().enumerate().flat_map(|(queue, column)| column.iter().enumerate().map(move |(index, submission)| (submission.wave, queue, index))).collect::<Vec<_>>();
        order.sort_unstable();
        let order = order.into_iter().map(|(_, queue, index)| (QUEUES[queue], index)).collect();

        SubmissionPlan {
            submissions,
            order,
        }
    }
}

// Folds adjacent candidates into submissions.
fn compact(mut raw: Vec<[Option<Submission>; 3]>, wave_count: usize) -> ([Vec<Submission>; 3], Vec<[usize; 3]>) {
    let mut submissions: [Vec<Submission>; 3] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    let mut submission_of_wave = vec![[usize::MAX; 3]; wave_count];

    for queue in 0..3 {
        let mut open: Option<Submission> = None;
        let mut previous_was_target = false;

        for wave in 0..wave_count {
            let Some(raw) = raw[wave][queue].take() else {
                continue;
            };
            let is_target = raw.is_wait_target;
            let terminal = raw.terminal;
            let joins = open.is_some() && !previous_was_target && raw.is_plain() && !terminal;

            if joins {
                submission_of_wave[wave][queue] = submissions[queue].len();
                open.as_mut().expect("open exists").absorb(raw);
            } else {
                if let Some(done) = open.take() {
                    submissions[queue].push(done);
                }
                submission_of_wave[wave][queue] = submissions[queue].len();
                if terminal {
                    submissions[queue].push(raw);
                } else {
                    open = Some(raw);
                }
            }
            previous_was_target = is_target;
        }

        if let Some(done) = open.take() {
            submissions[queue].push(done);
        }
    }

    (submissions, submission_of_wave)
}

// Rewrites source waves to submission indices.
fn resolve_waits(submissions: &mut [Vec<Submission>; 3], submission_of_wave: &[[usize; 3]]) {
    for column in submissions.iter_mut() {
        for submission in column.iter_mut() {
            let mut resolved = Vec::with_capacity(submission.waits.len());
            for wait in submission.waits.drain(..) {
                let source = submission_of_wave[wait.source][qi(wait.queue)];
                debug_assert!(source != usize::MAX, "wait references a wave with no submission");
                push_exact(&mut resolved, wait.queue, source, wait.stage);
            }
            submission.waits = resolved;
        }
    }
}

fn push_exact(waits: &mut Vec<Wait>, queue: QueueType, source: usize, stage: PipelineStage) {
    match waits.iter_mut().find(|wait| wait.queue == queue && wait.source == source) {
        Some(existing) => existing.stage |= stage,
        None => waits.push(Wait {
            queue,
            source,
            stage,
        }),
    }
}

fn push_slot(waits: &mut Vec<SlotWait>, queue: QueueType, stage: PipelineStage) {
    match waits.iter_mut().find(|wait| wait.queue == queue) {
        Some(existing) => existing.stage |= stage,
        None => waits.push(SlotWait {
            queue,
            stage,
        }),
    }
}
