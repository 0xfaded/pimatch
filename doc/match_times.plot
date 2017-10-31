set terminal png font "Droid Sans Mono" size 800,400
set output 'match_times.png'

set xlabel "√comparisons"
set xrange [200:2050]
set ylabel "time"
set ytics ("0ms" 0, "10ms" 10, "20ms" 20, "30ms" 30, "40ms" 40, "50ms" 50, "60ms" 60, "70ms" 70)

set key left

set style data histogram
set style histogram cluster gap 3

set style fill solid border rgb "black"
set auto x
set yrange [0:*]
plot 'match_times' using 2:xtic(1) title col, \
        '' using 3:xtic(1) title col, \
        '' using 4:xtic(1) title col, \
        '' using 5:xtic(1) title col, \
        '' using 6:xtic(1) title col, \
        '' using 7:xtic(1) title col, \
        '' using 8:xtic(1) title col, \
        '' using 9:xtic(1) title col, \
        '' using 10:xtic(1) title col, \
        '' using 11:xtic(1) title col


