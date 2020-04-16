use std::process;

fn main() {
    //let root = String::from("/work/ku0598/k202107/git/mistral-job-evaluation/data/eval_20200117/");
    //let root = String::from("/home/joobog/dkrz/git/job_io_datasets/datasets");
    let root = String::from("/work/ku0598/k202107/git/mistral-io-datasets/datasets");
    //let dset_fn = format!("{}/coding_job_abs_mode_False.csv", root);
    let dset_fn = format!("{}/codings.csv", root);
    let output_fn = format!("{}/codings_similarity.csv", root);

    println!("{}", dset_fn);

    let cfg = run::Config{
        dataset_fn: dset_fn,
        output_fn: output_fn,
        nrows: 1_000_000,
        batch_size: 1000,
        min_similarity: 0.7,
        n_workers: 72,
    };

    if let Err(e) = run::run(cfg) {
       eprintln!("Error occured in run: {}", e);
       process::exit(1);
    }
}
