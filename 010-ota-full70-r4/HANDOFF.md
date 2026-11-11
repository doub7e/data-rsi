# COMPLETED — 2026-11-10

All requested010 stages complete.27boardrunsverified; noota-r4RunAIjobs remain. Results inREADME.md andruns/comparison/REPORT.md. Guided underperformsUniform; all70tasksfeedback-used,one training seed. Do not restart services or rerun completed trials. Previous handoff below is historical.

# ACTIVE: full70×4 resampling experiment

## Latest recovery update

At2026-11-08~15:50UTC, RunAI replaced inference pod0-0 with0-1, which is Pending. The original RunAI job/queue was retained. The stale CPUclient was stopped, awaited to exit, and only its32owned orphan sandboxes cleaned.66 completed probe records are hash-verified unchanged. Controller/guard are active again.

New controller detects missing/non-running/replaced inference pods and pauses dependent clients immediately, retaining queued replacements. Endpoint storespodUID; model+health readiness checked. Previously-ready sameUID unhealthy6consecutivechecks gets bounded2serverrestarts. Pending replacements are retained by resourceguard. Rollout resume preloads ALL cached records before dispatch, so progress does not fall; valid finished raw verifier outcomes without markers are postprocessed before any Harbor resume, never rerolled. Unit checks passed. Monitor by workload label/currentpod, not hardcoded0-0.


User explicitly requests: evaluate first-epoch model on all70 tasks4times, use these280outcomes for probe with missing verifier score0, reselect environments and regenerate teacher data then continue training1epoch, evaluate newGuided70×4, supplement existingUniform from1to4repeats. Complete the whole workflow, not just submit. User prior preference: persist with bounded sleep and monitoring. Do not spawn subagents. Use experiment-board-recording; all results in this directory. Old009 is complete and immutable.

## Setup and current execution

- New010-ota-full70-r4 registered and board identity/denominators verified. First stage probe280new. Uniform shows70/280 reused and210pending. NewGuided280fresh. Total770new student rollouts plus728new teacher rollouts.
- Services data-rsi-ota-r4-1108.service (controller.py) and data-rsi-ota-r4-guard-1108.service (resource_guard.py) ACTIVE. Logs runs/study/controller.log/resource-guard.log. Job prefix ota-r4-...-1108. First server ota-r4-serve-epoch1-1108 submitted4H200; last observed replacement pod0-1 Pending at16:28UTC.66formal outcomes completed. Check actual state before acting.
- Source009 first-epoch and Uniform checkpoints are read-only references in configs/protocol.json:model_paths. Teacher unchanged pinnedQwen3.8-27B. Do not retrain first epoch orUniform, and do not reuse oldGuided checkpoint.
- First teacher728data copied as immutable hardlinks for candidate gradients. Uniform's original70raw job folders and SFT exports imported as hardlinks; copied derived complete records atomically replaced with newscore policy/repeat1/provenance. Source009records verified unchanged. Oldraw files must never be edited. Cache guard test proves reusedUniform trials do not start Harbor.
- scripts/common.py defines ORIGIN009, TASKS=ORIGIN/tasks/native. Existing798preparedtaskimages reused; no image build. Actual tasks/seeds are read-only source refs; newtrial results in010.
- Same pinned train/worker/vLLM images as009, student eager+disablecustomAR. uid223974/gid11166, PVC/scratch.4H200serving/training, max8total requestedGPUs,32sandboxes, one inference server at a time. Keepqueued, don't fallback/changeconditions withoutusersteering.
- KUBECONFIG tokenonly under workspace .codex/secrets/native-runai-kubeconfig.json; only existing native-auth publisher refreshes. Do not touch siblingjobs or otherauthconsumers.

## Frozen method

- All70tasks used for feedback; NO independent heldout set. Repeat-major item order, e0001–e0280. Probe seeds1800000000+index*100000; postUniform/newGuided seeds1900000000+index*100000, identical across the two finalmodels. OldUniform first70seeds match exactly. Probe/post seeds deliberatelydistinct. TeacherGuided1600000000+index*100000. Environment samplingseed remains20261109 aspreviousprotocol.
- Probe4/task, rawverifierreward or0ifmissing. Preserve raw_score/null,exception,imputed_zero. Same0rule for all3models and reusedUniformrepeat1. Bounded infrarecovery remains; after exhausted retries a realterminalattempt can score0, never fabricate an unexecuted trial.
- Dropconstant-score questions, noadaptive replacement. Actual LF/Liger loss, sameencoder/mask_historyfalse/train_on_promptfalse/cutoff32768/no packing, BF16model forward andFP32gradient accumulation. Numericalcontract reusedfrom009calibration on exactsamecheckpoint/stack. Never uselegacyFP32autocast.
- Up to4full-parameter probe-gradient shards (probe_gradient_shard.py, N=4), globaleligiblequestionnormalization, CPUmerge. No70separate32GBvectors. Candidate728dots2×1H200workers at samefirst-epochcheckpoint. Same signed dot→clippedzscore softmax with10%uniformfloor. Fresh teacher728withreplacement. Allmodeloutcomes trained; no success-onlyfilter. NewGuided starts009epoch1 with resetoptimizer/scheduler sameasUniform;1epoch23-ishsteps dependingactualSFTrowcount.
- Aggregate.py compares epoch1,Uniform,Guided all280outcomes. Mean/successcount perquestion over4, pairedtaskbootstrap over70questions (not280independentquestions); postmodels paired seeds. Explicitfeedback-use,rawcoverage,missing0counts; plots/report/CSV produced in comparisonrun.

## Validation already done

scripts/check_protocol.py passed:70×4plans,seeds,cache-no-reroll,original70recordhashes,missing0denominator,samplingprobabilitymath, AST-equal trainencoder/loss andBF16productioncalls,syntaxallfiles. Per-task reportingtoytest passed. NumPy bootstrap/reporting check passed on first allocated inference worker (runs/study/bootstrap-unit-check.json). No CUDA onjump host.

Allscripts for remainingDAG are implemented, including aggregate.py. Firstserver can run while reviewing downstreamcode. 66 formal outcomes at16:28UTC; check files/jobs forlatest.

## Remaining work

1. Monitor firstserver readiness/generation-smoke and probe280. Verify actualcodepaths/tasksseeds andearlyoutputs; inspect unresolved-trials immediately, healthypeers continue.
2. Onprobecomplete, gradient plan→up to4GPUshards→CPUmerge→candidate2GPUs. Uniformsupplement automaticallyoverlapsusingone4H200server afterprobe frees it.
3. Inspect allcandidate scores/probabilities and selectedenvmultiplicities; newteacher728→newGuided1epoch→eval280. Follow anyfailures without rerolling validterminalmodel failures.
4. Inspect per-task4repeatcounts/imputed flags,source009immutability,compare toepoch1andUniform,CIbootstrap unit,plots,actualboardAPIallruns andtraining/evalviewers. Clear onlyownjobs aftercomplete. UpdateREADME/HANDOFFfinal.

Usefunctions.exec/tools.exec_command(login=false,sandbox_permissions=require_escalated) becausedefaultbwrapfails. Longmonitor waits≤60s. Priorautomaticreview issues are not newpermissionsrequirements; latestuser explicitlyauthorizes missing-as-zero. If a newreviewrejectsaction, followitsprocedure and don't bypass. No pendingapproval at present.

##16:31UTC audit
Controller terminal-server recovery now uses pause_dependents (wait for client exit before sandbox cleanup), matching pod relocation recovery. Probe eligibility plan explicitly asserts all derived scores valid, retains missing-zero questions and excludes only constant derived scores; formula text corrected from stale009 wording. Missing-zero mixed/constant toy check and invalid-derived-score assertion passed. No-signal metadata includes excluded question count for final report. Actual board e0015 completed trajectory/segment content checked successfully; generic helper selected an unfinished non-error trial with no segment and reported failure, preserved separately. Current CPU client absent while awaiting queued GPU replacement; board stale warning expected for absent paused client. No formal outcome changed.

##17:03UTC recovery resumed
Replacement inferencepod0-1 nowRunning, newendpoint172.16.116.112:8000/v1;41request concurrency smoke passed. CPUprobe client restarted17:01UTC, preloaded66/280cachedoutcomes, all66hashesrevalidatedunchanged;32native sandboxesRunning, unresolved0. Controller/guard/authpublisheractive. Boundedforegroundmonitor session16606 (72×50s) observesprogress, stop/finishbeforehandoff.

##17:08UTC current
First new completed result after recovery accepted:67/280, unresolved0. Actual board live stateRunning/staleFalse and denominator280 verified. All66pre-recovery complete hashesunchanged. Foregroundmonitor16606 stopped intentionally; no foregroundmonitor remains. Durablecontroller+guard keep full pipeline running. Experiment is NOT complete; all laterstages stillpending.

##2026-11-09 authorized quota cleanup and recovery
User explicitly approved first3cleanupgroups. Deleted4merged010gradientshards,3old009checkpoint23dirs,002checkpoint700/735,003checkpoint70/77:869.727918GiB allocated. Fullaudit runs/study/storage-cleanup-20261109; allfinalmodelshards validatedheaders/size and preservedmetadata. Originalfirstepoch/Uniform modelroots intact. Old per-step metadata archived. Records linked in affected board runs and verifiedactualAPI. Nativeauth recovered automatically.
010training oldpodFailed quota duringcheckpoint10 save; incomplete94GiBcheckpoint preserved by rename into runs/sft-guided/recovery/quota-failed-1109/checkpoint, not deleted. No trainer_state.json, so cannot resumeexactly. Oldclusterpod/log archived,failedjobremoved. Newtraining job ota-r4-sft-guided-r1-1109 submitted same data/model/config/seed; controllerNAMES map updated. Controllerinitialrestart submittedsuccessfully but stdout quota error occurred; explicitnewfile/log fsyncpassed andcontrollerrestartedagain, reuses alreadyqueuedjob. Serviceskeeporiginalunitnames. Check lateststatusbeforeacting. FullGuidedeval280 stillpending; Uniform280 andteacher728completed.
