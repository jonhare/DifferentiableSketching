OUT=results/vancouver_crs
imageopt --loss LPIPSLoss --net vgg --invert --seed 1234 --width 300 --crs 500 --init-sigma2 1.0 --final-sigma2 1.0 --iters 500 --lr 0.01 --init-raster $OUT/init.png --final-raster $OUT/final.png --init-pdf $OUT/init.pdf --final-pdf $OUT/final.pdf --snapshots-path $OUT data/vancouver.jpg --snapshots-steps 100 --colour
