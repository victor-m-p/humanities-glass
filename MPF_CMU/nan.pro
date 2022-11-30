good=[[0, 0.719342, 0.372065, 0.719342], [64.0, 0.451923, 0.372065, 0.719342], [128.0, 0.409337, 0.372065, 0.719342], [256.0, 0.297186, 0.372065, 0.719342], [512.0, 0.214251, 0.372065, 0.719342], [768.0, 0.172086, 0.372065, 0.719342], [1024.0, 0.14055, 0.372065, 0.719342]]
bad=[[0, 0.719342, 0.372065, 0.719342], [64.0, 0.509556, 0.372065, 0.719342], [128.0, 0.494364, 0.372065, 0.719342], [256.0, 0.428575, 0.372065, 0.719342], [512.0, 0.432226, 0.372065, 0.719342], [768.0, 0.396059, 0.372065, 0.719342], [1024.0, 0.396002, 0.372065, 0.719342]]

set_plot, 'ps'
device, filename="PLOTS/nan.eps"

plot, good[0,*], good[1,*], psym=-2, yrange=[0,max(bad[1,*])], xrange=[0,1024], xstyle=1, xtitle='Additional partial observations', ytitle='KL from True Distribution'
oplot, bad[0,*], bad[1,*], psym=-2

oplot, good[0,*], good[3,*]
xyouts, 600, good[3,*]-0.025, 'KL for 128 complete observations'

oplot, good[0,*], good[2,*]
xyouts, 600, good[2,*]-0.025, 'KL for 256 complete observations'

xyouts, 600, bad[1,where(bad[0,*] eq 512)]+0.025, 'Naieve completion strategy'

xyouts, 600, good[1,where(bad[0,*] eq 512)]-0.05, 'Bayesian estimation'

device, /close_file
set_plot, 'x'

stop
end