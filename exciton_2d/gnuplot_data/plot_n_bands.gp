set terminal pdfcairo enhanced font 'Arial,40' size 16,12
set output 'output_figures/n_colored_band_structure_a016_v00.1.pdf'
set lmargin 10
set rmargin 10
set tmargin 3
set bmargin 3
set title 'Exciton Band Structure (a₀=16.0, V₀=0.1)' font 'Arial,16' textcolor rgb '#e0e0e0'
set xlabel 'Wave vector' textcolor rgb '#e0e0e0'
set ylabel 'Energy (a.u.)' textcolor rgb '#e0e0e0'
set xrange [0:0.6703792653336215]
set yrange [-0.22007252277083827:0.06902396364173993]
set palette rgbformulae 22,13,-22
set colorbox vertical user origin 0.88,0.1 size 0.03,0.8
set cbrange [0.002565010926409411:2.99497111059698]
set cblabel 'Principal Quantum Number (n)' offset 2,0
set cbtics 1
set xtics ('Γ' 0, 'X' 0.19634954084936207, 'M' 0.3926990816987237, 'Γ' 0.6703792653336215)
set arrow from 0,-0.22007252277083827 to 0,0.06902396364173993 nohead lc rgb '#888888' lw 6
set arrow from 0.19634954084936207,-0.22007252277083827 to 0.19634954084936207,0.06902396364173993 nohead lc rgb '#888888' lw 6
set arrow from 0.3926990816987237,-0.22007252277083827 to 0.3926990816987237,0.06902396364173993 nohead lc rgb '#888888' lw 6
set arrow from 0.6703792653336215,-0.22007252277083827 to 0.6703792653336215,0.06902396364173993 nohead lc rgb '#888888' lw 6
plot 'gnuplot_data/n_band_0.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_1.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_2.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_3.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_4.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_5.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_6.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_7.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_8.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_9.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_10.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_11.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_12.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_13.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_14.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_15.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_16.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_17.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_18.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_19.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_20.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_21.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_22.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_23.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_24.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_25.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_26.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_27.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_28.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_29.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_30.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_31.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_32.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_33.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_34.dat' using 1:2:3 with lines lw 8 lc palette notitle
