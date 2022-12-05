ent=7.059176
;; close match to the observed

openr, 1, 'TEST/test_params.dat_probs.dat'
n=2l^20
true=dblarr(3, n)
readf, 1, true
true=reform(true[2,*],n)
close, 1

openr, 1, 'TEST/test_data.dat_params.dat_probs_CV.dat'
n=2l^20
cv=dblarr(3, n)
readf, 1, cv
cv=reform(cv[2,*],n)
close, 1

openr, 1, 'TEST/test_data.dat_params.dat_probs.dat'
n=2l^20
no=dblarr(3, n)
readf, 1, no
no=reform(no[2,*],n)
close, 1

w=where(no lt 2e-20)
no[w]=2e-20

c = FINDGEN(17) * (!PI*2/16.)  
s=0.2
USERSYM, s*COS(c), s*SIN(c), /FILL

set_plot, 'ps'
device, filename='PLOTS/sample_fit_new.eps', SET_FONT='Helvetica', /TT_FONT, bits_per_pixel=8
colors=[cgColor('Dodger Blue', 1), cgColor('Sea Green', 2), cgColor('Crimson', 3), cgColor('Dark Goldenrod', 4), cgColor('Charcoal', 5), cgColor('Light Cyan', 6)]

;xrange=[1e-8, 1], yrange=[1e-20, 1]
plot, true, cv, /xlog, /ylog, xrange=[1e-8, 1], yrange=[1e-20, 1], psym=4, xtitle='True Probabilities', ytitle='Inferred Probabilities', /nodata, font=1, xthick=2, ythick=2, charthick=2, charsize=1.5
oplot, true, no, psym=8, color=3

oplot, true, no, psym=7, color=3, SymSize=0.25
oplot, true, cv, psym=8, color=1

; xyouts, 2e-3, 1e-7, 'No constraint: KL '+strn(kl_nosparsity/alog(2))+' bits', color=3, font=1
; xyouts, 2e-3, 1e-8, 'Sparsity: KL '+strn(kl_neg0/alog(2))+' bits', color=1, font=1

oplot, [1d-20,1], [1d-20,1], thick=3

device, /close_file
set_plot, 'x'

stop
end