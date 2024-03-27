import arguments
import logging
import mlperf_loadgen as lg
import os
import sys
import coco
from sut import OfflineSUT, ServerSUT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Stable-diffusion")




scenario_map = {
    "SingleStream": lg.TestScenario.SingleStream,
    "Offline": lg.TestScenario.Offline,
    "Server": lg.TestScenario.Server,
}


def main():
    args = arguments.parse_args()

    print("self.workers_per_proc & total_sample_count : ",args.workers_per_proc, args.total_sample_count)

    if args.scenario == "Offline":
        sut = OfflineSUT(
            dataset_path=args.dataset_path,
            num_proc=args.num_proc, 
            cpus_per_proc=args.cpus_per_proc, 
            total_sample_count=args.total_sample_count,
            dtype=args.dtype,
            device_type=args.device,
            scenario=args.scenario,
            workers_per_proc=args.workers_per_proc,
            initial_core=args.cores_offset,
            args=args,
        )
    else:
        sut = ServerSUT(
            dataset_path=args.dataset_path,
            num_proc=args.num_proc, 
            cpus_per_proc=args.cpus_per_proc, 
            total_sample_count=args.total_sample_count,
            dtype=args.dtype,
            device_type=args.device,
            scenario=args.scenario,
            workers_per_proc=args.workers_per_proc,
            initial_core=args.cores_offset,
            args=args,
        )
    sut.startSUT()
    lg_qsl = lg.ConstructQSL(
        sut.dataset.count,
        sut.dataset.perf_count,
        sut.load_query_samples,
        sut.unload_query_samples,
    )
    lg_sut = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    # set cfg
    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario]
    settings.FromConfig(args.mlperf_conf, "stable-diffusion-xl", args.scenario)
    settings.FromConfig(args.user_conf, "stable-diffusion-xl", args.scenario)
    settings.mode = (
        lg.TestMode.AccuracyOnly if args.accuracy else lg.TestMode.PerformanceOnly
    )
    # set log
    os.makedirs(args.log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = False

    result_dict = {"scenario": str(settings.scenario)}

    # run benchmark
    log.info("==> Running loadgen test")
    lg.StartTestWithLogSettings(lg_sut, lg_qsl, settings, log_settings, args.audit_conf)
    log.info("Test completed")


    '''if args.accuracy:
        result_dict = {"scenario": str(args.scenario)}
        sut.postprocess_accuracy(result_dict)
        final_results["accuracy_results"] = result_dict
        #
        # write final results
        #
        if args.output:
            with open("results.json", "w") as f:
                json.dump(final_results, f, sort_keys=True, indent=4)'''

    sut.stop_sut()

    lg.DestroyQSL(lg_qsl)
    lg.DestroySUT(lg_sut)
    log.info("Done!")


if __name__ == "__main__":
    main()
