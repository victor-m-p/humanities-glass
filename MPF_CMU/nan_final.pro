full_data=[[0, 0.513076], [64, 0.408739], [128, 0.329577], [256, 0.246484], [512, 0.169543], [1024, 0.104252]]
nan_data=[[0, 0.513076], [64, 0.476982], [128, 0.431512], [256, 0.350478], [512, 0.30196], [1024, 0.220251]]
bad_data=[[0, 0.513031], [64, 0.541769], [128, 0.497583], [256, 0.485234], [512, 0.479655], [1024, 0.561929]]

full_data=[[0, 0.60265], [64, 0.45424], [128, 0.366766], [256, 0.254192], [512, 0.17044], [1024, 0.10341]]
nan_data=[[0, 0.60265], [64, 0.498995], [128, 0.417302], [256, 0.326236], [512, 0.229564], [1024, 0.166233]]
bad_data=[[0, 0.60265], [64, 0.568289], [128, 0.569504], [256, 0.565376], [512, 0.581938], [1024, 0.628065]]

set_plot, 'ps'
device, filename='PLOTS/nan_example.eps', SET_FONT='Helvetica', /TT_FONT, bits_per_pixel=8
colors=[cgColor('Dodger Blue', 1), cgColor('Sea Green', 2), cgColor('Crimson', 3), cgColor('Dark Goldenrod', 4), cgColor('Charcoal', 5), cgColor('Light Cyan', 6)]

;xrange=[1e-8, 1], yrange=[1e-20, 1]
plot, full_data[0,*], full_data[1,*], xrange=[0,1024], xstyle=1, psym=4, xtitle='Amount of additional data', ytitle='KL(Estimated, Actual)', /nodata, font=1, xthick=2, ythick=2, charthick=2, charsize=1.5

oplot, full_data[0,*], full_data[1,*], psym=-4, color=4, thick=3

oplot, nan_data[0,*], nan_data[1,*], psym=-4, color=1, thick=3
oplot, bad_data[0,*], bad_data[1,*], psym=-4, color=3, thick=3

xyouts, 600, 0.1, 'Ideal (Complete Data)', color=4, font=1
xyouts, 600, 0.225, 'NA with Partial-MPF', color=1, font=1
xyouts, 600, 0.55, 'NA with Naive strategy', color=3, font=1


device, /close_file
set_plot, 'x'

stop

end