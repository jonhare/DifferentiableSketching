import glob

genargs = "--invert --seed 1234 --width 300 --iters 500 --lr 0.01 --init-raster {out}/init.png --final-raster {out}/final.png --final-pdf {out}/final.pdf {infile}"

losses = [("--loss LPIPSLoss --net vgg", "lpipsvgg"),
          # ("--loss LPIPSLoss --net alex", "lpipsalex"),
          ("--loss MSELoss", "mse")]
types = [("--lines 500", "500lines"),
         ("--crs 500", "500crs"),
         ("--points 500", "500pts")]
widths = [("--init-sigma2 1.0 --final-sigma2 1.0", "fixed"),
          ("--init-sigma2 15.0 --opt-sigma2 --sigma2-lr 0.00001", "learned")]
cols = [("", "bw"),
        ("--colour", "col")]

print("#!/bin/sh")
for infile in glob.glob("data/extras/*.JPG"):
    outbase = "results/extras/" + infile[infile.rindex('/') + 1:-4] + '/'

    for loss, lossname in losses:
        for typ, typname in types:
            for width, widthname in widths:
                for col, colname in cols:
                    out = outbase + f"{lossname}-{typname}-{widthname}-{colname}"

                    cmd = f"imageopt {loss} {typ} {width} {col} " + genargs.format(out=out, infile=infile)
                    print(cmd)
