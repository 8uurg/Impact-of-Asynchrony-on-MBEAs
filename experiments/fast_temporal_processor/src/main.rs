mod quantile_heap;

use std::fs::File;

use itertools::izip;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use anyhow::Result;
use polars::prelude::*;

use clap::Parser;
use quantile_heap::QuantileKind;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    input: String,
}

fn quantile_map(
    q: f64,
    qk: QuantileKind,
    k: &mut Series,
    m: &mut Series,
    v: &mut Series,
) -> Result<Series, PolarsError> {
    let mut acc = quantile_heap::DualHeap::new(q);

    let kc = k.u64().unwrap().into_iter();
    let mc = m.bool().unwrap().into_iter();
    let vc = v.f64().unwrap().into_iter();

    let i = izip!(kc, mc, vc).map(move |(key, mode, value)| {
        if !mode.unwrap() {
            // Insert!
            acc.add(key.unwrap().try_into().unwrap(), value.unwrap());
        } else {
            acc.remove(key.unwrap().try_into().unwrap());
        }
        acc.quantile(&qk)
    });

    Ok(Float64Chunked::from_iter(i).into_series())
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Let us assume this dataframe has the following schema:
    // c (u64) - Category index
    // t (f64) - time
    // o (u64) - ordinal - unique (even for duplicate time values), in order with time.
    // k (u64) - key - unique per in/out pair.
    // m (bool) - mode - insertion / start, or out?
    // v (f64) - value over which quantiles are computed.
    // let ldf = LazyFrame::scan_ipc(
    //     args.input,
    //     ScanArgsIpc {
    //         cache: true,
    //         rechunk: true,
    //         ..Default::default()
    //     },
    // )?;
    // println!("Loading...");
    let file = File::open(args.input)?;
    let df = IpcReader::new(file).finish()?;
    let ldf = df.lazy();
    // let ldf = LazyFrame::(
    //     args.input,
    //     ScanArgsIpc {
    //         cache: true,
    //         rechunk: true,
    //         ..Default::default()
    //     },
    // )?;

    // println!("Setting up...");
    let quantiles_no_dedup = ldf
        // .sort_by_exprs([col("c"), col("t"), col("ord")], [false, false, false].to_vec())
        .groupby([col("c")])
        .agg([
            all().exclude(["c"]),
            apply_multiple(
                |x| match x {
                    [k, m, v] => quantile_map(0.05, QuantileKind::LinearInterp, k, m, v),
                    _ => unreachable!(),
                },
                [col("k"), col("m"), col("v")],
                GetOutput::from_type(Float64Type::get_dtype()),
            )
            .alias("q0.05"),
            apply_multiple(
                |x| match x {
                    [k, m, v] => quantile_map(0.50, QuantileKind::LinearInterp, k, m, v),
                    _ => unreachable!(),
                },
                [col("k"), col("m"), col("v")],
                GetOutput::from_type(Float64Type::get_dtype()),
            )
            .alias("q0.5"),
            apply_multiple(
                |x| match x {
                    [k, m, v] => quantile_map(0.95, QuantileKind::LinearInterp, k, m, v),
                    _ => unreachable!(),
                },
                [col("k"), col("m"), col("v")],
                GetOutput::from_type(Float64Type::get_dtype()),
            )
            .alias("q0.95"),
        ]).explode([all().exclude(["c"])]);

    let f = quantiles_no_dedup;
    
    // println!("Processing..");
    let mut df = f.collect()?;
    
    // println!("Writing...");
    // Perform query and write to file
    let file = File::create("out.ipc")?;
    IpcWriter::new(file).with_compression(Some(IpcCompression::ZSTD)).finish(&mut df)?;
    Ok(())
}
