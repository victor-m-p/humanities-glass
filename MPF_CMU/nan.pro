good=[[0, 0.617184, 0.351919, 0.209386, 0.617184], [64.0, 0.525828, 0.351919, 0.209386, 0.617184], [128.0, 0.473472, 0.351919, 0.209386, 0.617184], [256.0, 0.407867, 0.351919, 0.209386, 0.617184], [512.0, 0.274963, 0.351919, 0.209386, 0.617184], [768.0, 0.23741, 0.351919, 0.209386, 0.617184], [1024.0, 0.200955, 0.351919, 0.209386, 0.617184]]
bad=[[0, 0.617184, 0.351919, 0.209386, 0.617184], [64.0, 0.59462, 0.351919, 0.209386, 0.617184], [128.0, 0.56937, 0.351919, 0.209386, 0.617184], [256.0, 0.615963, 0.351919, 0.209386, 0.617184], [512.0, 0.624162, 0.351919, 0.209386, 0.617184], [768.0, 0.659886, 0.351919, 0.209386, 0.617184], [1024.0, 0.665563, 0.351919, 0.209386, 0.617184]]

set_plot, 'ps'
device, filename="PLOTS/nan.eps"

plot, good[0,*], good[1,*], psym=-2, yrange=[0,max(bad[1,*])], xrange=[0,1024], xstyle=1, xtitle='Additional partial observations', ytitle='KL from True Distribution'
oplot, bad[0,*], bad[1,*], psym=-2

oplot, good[0,*], good[4,*]
xyouts, 600, good[3,*]-0.025, 'KL for 128 complete observations'

oplot, good[0,*], good[2,*]
xyouts, 600, good[2,*]-0.025, 'KL for 256 complete observations'

oplot, good[0,*], good[3,*]
xyouts, 600, good[2,*]-0.025, 'KL for 512 complete observations'

xyouts, 600, bad[1,where(bad[0,*] eq 512)]+0.025, 'Naieve completion strategy'

xyouts, 600, good[1,where(bad[0,*] eq 512)]-0.05, 'Bayesian estimation'

device, /close_file
set_plot, 'x'

stop
end