set terminal pngcairo enhanced font 'Arial,40' size 2400,1800
set output 'output_figures/n_colored_band_structure_a09_v00.05.png'
set lmargin 10
set rmargin 10
set tmargin 3
set bmargin 3
set title 'Exciton Band Structure (a₀=9.0, V₀=0.1)' font 'Arial,16'
set xlabel 'Wave vector'
set ylabel 'Energy (a.u.)'
set xrange [0:1.1917853605931057]
set yrange [-0.4101480668290248:0.033757702359355504]
set palette rgbformulae 22,13,-22
set colorbox vertical user origin 0.88,0.1 size 0.03,0.8
set cbrange [0.004787513142060862:2]
set cblabel 'Principal Quantum Number (n)' offset 2,0
set cbtics 1
set xtics ('Γ' 0, 'X' 0.3490658503988659, 'M' 0.698131700797733, 'Γ' 1.1917853605931057)
set arrow from 0,-0.4101480668290248 to 0,0.033757702359355504 nohead lc rgb '#888888' lw 6
set arrow from 0.3490658503988659,-0.4101480668290248 to 0.3490658503988659,0.033757702359355504 nohead lc rgb '#888888' lw 6
set arrow from 0.698131700797733,-0.4101480668290248 to 0.698131700797733,0.033757702359355504 nohead lc rgb '#888888' lw 6
set arrow from 1.1917853605931057,-0.4101480668290248 to 1.1917853605931057,0.033757702359355504 nohead lc rgb '#888888' lw 6
plot 'gnuplot_data/n_band_0.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_1.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_2.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_3.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_4.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_5.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_6.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_7.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_8.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_9.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_10.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_11.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_12.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_13.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_14.dat' using 1:2:3 with lines lw 8 lc palette notitle, 'gnuplot_data/n_band_15.dat' using 1:2:3 with lines lw 8 lc palette notitle
