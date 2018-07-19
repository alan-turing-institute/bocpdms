#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 18:30:21 2018

@author: jeremiasknoblauch

Description: Compute #Params for Turner vs us
"""

import numpy as np
import matplotlib.pyplot as plt


#PF data
num_param_Turner_AR1 = 30 * ( 1 ) #only one AR-coef + one variance/kernel term!
num_param_Turner_AR5 = 30 * (5)

us_raw = [2, 6, 3, (6+3+2+1), (6+3+2+1), 3, 3] #number of linear predictors per location
num_param_us = 0
for p in us_raw:
    p = 30 * p #number of predictors
    num_param_us = num_param_us + (2 + (p + (p*(p+1)*0.5)) ) #2 = a, b posteriors, p = regr.,p*(p+1)*0.5 = cov mat 

#computation time for Turner = 3h, 21min, 17s (AR1) and 5h, 49min, 23s (AR5)
Turner_duration_AR1 = (3*60 + 21)*60 + 17
Turner_duration_AR5 = (5*60 + 49)*60 + 23

#computation time ICML EvT object (retrievable as Evt.results[1] from 
#   paper_pictures_30Portfolios_... )
us_duration = 239287.467205
T=8707

#comparison for the number of params
Turner_dur_per_param = Turner_duration_AR1/num_param_Turner_AR1
Turner_dur_per_param_AR5 = Turner_duration_AR5/num_param_Turner_AR5
us_dur_per_param = us_duration/num_param_us

print("Turner's ARGPCP method needs", Turner_dur_per_param, 
      "seconds per param")
print("BOCPDMS with", num_param_us, "parameters needs", 
      us_dur_per_param, "seconds per param")

print("Turner's ARGPCP method needs", Turner_dur_per_param/60, 
      "minutes per param")
print("BOCPDMS with", num_param_us, "parameters needs", 
      us_dur_per_param/60, "minutes per param")

print("In summary then, BOCPDMS is", Turner_dur_per_param/us_dur_per_param, 
      "times faster per parameter")

print("For AR(5) ARGPCP, BOCPDMS is", Turner_dur_per_param_AR5/us_dur_per_param, 
      "times faster per parameter")

print("AR(1) ARGPCP takes", Turner_duration_AR1/T, 
      "sec. per observation,", 
      "BOCPDMS takes", us_duration/T)



time = 239287
T=8707
time_GP = 12077
num_param = 0
num_param_GP = 30 #Ryan Turner runs an AR(1) in his code, and AR-coefs are the only
                  #param in that model which are inferred in Bayesian manner

plist = [2, 6, 3, (6+3+2+1), (6+3+2+1), 3, 3]
for p in plist:
    p=30*p #dimensionality = 30
    num_param = num_param + (2 + (p + (p*(p+1)*0.5)) )
print("-------------------------------------")
print("30 PORTFOLIOS")
print("time per obs", time/T)
print("time / param = ", time/num_param)
print("time per obs for GP", time_GP/T)
print("time / param for GP", time_GP/num_param_GP)
print("time per model BOCPDMS", time/len(plist))
print("time per model GP", time_GP)
print("-------------------------------------")


time = 1574
T= 8046
time_GP = 284
num_param = 0
num_param_GP = 2

plist = [2,3,4,5,6,7,8,9,10,11]
num_param=0
for p in plist:
    num_param = num_param + (2 + (p + (p*(p+1)*0.5)) )
print("-------------------------------------")
print("SNOW")
print("time per obs", time/T)
print("time / param = ", time/num_param)
print("time per obs for GP", time_GP/T)
print("time / param for GP", time_GP/num_param_GP)
print("time per model BOCPDMS", time/len(plist))
print("time per model GP", time_GP)
print("-------------------------------------")



time = 1460
T= 1057
time_GP = 164
num_param = 0
num_param_GP = 7

plist = [6, 9, 15, 21, 27, 33, 39, 45, 6, 9, 12, 15, 18, 21, 24]
num_param=0
for p in plist:
    p = 3 * p #dimensionality = 3
    num_param = num_param + (2 + (p + (p*(p+1)*0.5)) ) 
print("-------------------------------------")
print("BEE")
print("time per obs", time/T)
print("time / param = ", time/num_param)
print("time per obs for GP", time_GP/T)
print("time / param for GP", time_GP/num_param_GP)
print("time per model BOCPDMS", time/len(plist))
print("time per model GP", time_GP)
print("-------------------------------------")



time = 12
T= 663
time_GP = 42
num_param = 0
num_param_GP = 2

plist = [4, 3, 2] 
num_param=0
for p in plist:
    num_param = num_param + (2 + (p + (p*(p+1)*0.5)) ) 
print("-------------------------------------")
print("NILE")
print("time per obs", time/T)
print("time / param = ", time/num_param)
print("time per obs for GP", time_GP/T)
print("time / param for GP", time_GP/num_param_GP)
print("time per model BOCPDMS", time/len(plist))
print("time per model GP", time_GP)
print("-------------------------------------")





#plot some of the complexity results
tpm = [
       np.array([4.0, 157.4,  97.33, 34183.857 ]), #time per model BOCPDMS
       np.array([42,  284,    164,  12077 ]) #time per model GP
      ]
tpp = [
       np.array([0.353, 4.254, 0.0392, 1.4806539239770062]), #time per param BOCPDMS
       np.array([21, 142.0, 23.43, 402.57]) #time per param GP
       ]
labels = ["Nile", "Snow", "Bee", "30PF"]

dir_ = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//Presentation//CompTime")

#plot how many times faster/slower BOCPDMS is
#height_ratio = [1,1]
fig, ax = plt.subplots(1, figsize=(8,2.5)) #(8,5)
plt.subplots_adjust(hspace = .1, left = None, bottom = None, right = None, 
                    top = None)

xlabsize, ylabsize, legendsize, ticksize = 15, 15, 13,15
textsize = 15

#plot tpm
relative_tpm = tpm[1]/tpm[0]
relative_tpp = tpp[1]/tpp[0]

#ax_array[0].plot([0,1,2,3],relative_tpm, marker = 'o',ms=7,
#                        linewidth = linewidths[0],
#                       linestyle = linestyles[0],
#                       color = linecolors[0])
ax.plot([0,1,2,3],relative_tpp, marker = 'o',ms=7,
                        linewidth = 3,
                       color = "navy")

#ax.set_xlabel("Data Set", size = xlabsize)

#ax.set_ylabel("BOCPDMS Speedup (per parameter)", size = ylabsize)
ax.set_ylabel("Speedup", size = ylabsize)

ax.set_ylim(-100,700)
ax.axhline(1, color = "gray", linestyle = ":")



ax.annotate('59.5', xy=(0, relative_tpp[0]), xytext=(0-0.1, 
            relative_tpp[0]+100), size = textsize
            )
ax.annotate('33.4', xy=(1, relative_tpp[1]), xytext=(1-0.1, 
            relative_tpp[1]+120), size = textsize
            )
ax.annotate('597.7', xy=(2, relative_tpp[2]), xytext=(2-0.1, 
            relative_tpp[2]-250), size = textsize
            )
ax.annotate('271.9', xy=(3, relative_tpp[3]), xytext=(3-0.2, 
            relative_tpp[3]+120), size = textsize
            )

#labels = ['ARGPCP', 'GPTSCP', 'NSGP','SSBVAR']
#plt.legend(handles, labels, prop = {'size':legendsize})
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(["Nile", "Snow", "Bee", "30PF"])
ax.set_yticks([1,250,500])
ax.set_yticklabels(["1", "250", "500"])

#ax_array[0].set_xticks([0,1,2,3],["Nile", "Snow", "Bee", "30PF"])
ax.tick_params(labelsize = ticksize)

fig.savefig(dir_ + "//CompTime.pdf",
            format = "pdf", dpi = 800)





