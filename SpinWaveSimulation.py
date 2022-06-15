#All imports go here
import time
import pstats
import cProfile
import copy
import math
import numpy
import scipy
import sys
import os
#import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import Process
from scipy import integrate
from numba import jit
from datetime import date

numpy.set_printoptions(threshold=sys.maxsize)

#Defining some constants
epsZero = 8.85 * 10 ** -12
muZero = 4 * numpy.pi * 10 ** -7

#Antenna Geometry
wsignal = 648 * 10 ** -9				#changed antenna widths to smaller antenna
wground = 324 * 10 ** -9
wgap = 334 * 10 ** -9
length_Antenna = 20 * 10 ** -6
distance_Antennas = 2.464 * 10 ** -6      #changed

#Setting up simulation points across Antenna
pts_ground = 26
del_width = wground / pts_ground
pts_total = int(numpy.ceil((wsignal + 2 * wground) / del_width))
pts_signal = pts_total - 2 * pts_ground
pts_max = max(pts_signal, pts_ground)
len_JJI_Z_vec = 12+pts_total+9


#Metal Characteristics
epsilonSi = 3.8				#changed added imaginary contribution
sigmaFM = 1.7 * 10 ** 7 
sigmaRu = 1 / (71 * 10 ** -9)
sigmaPt = 1 / (105 * 10 ** -9)
thicknessSi = 130 * 10 ** -9
thicknessFM = 20 * 10 ** -9
thicknessRu = 5 * 10 ** -9
thicknessPt = 5 * 10 ** -9
thicknessAl = 100 * 10 ** -9
resis_Al = 2.65 * 10 ** -8 #4 * 10 ** -8
var_Ys = resis_Al / thicknessAl
var_R1 = resis_Al / (thicknessAl * wsignal)
var_R2 = resis_Al / (thicknessAl * wground)
var_zc = 50

#Frequency range to test with Simulation
freq_lower = 2 * numpy.pi * 12 * 10 ** 9
freq_upper = 2 * numpy.pi * 24 * 10 ** 9
plot_pts_num = 100
if plot_pts_num > 1:
	freq_step = (freq_upper - freq_lower) / (plot_pts_num -1)
else:
	freq_step = 0


#Independent Variables
centralFreq = numpy.pi * (2 * 15 * 10 ** 9)
appliedH =	1000 * 79.57747#600 * 1.02 * 79.57747
linewidthSlope = 3.57 * 10 ** -9 * 79.57747 / (2*numpy.pi)
broadening = 0*7.96526 * 79.57747
gamma = 2 * numpy.pi * 3 * 10 ** 10 * 4 * numpy.pi * 10 ** -7
ampMs =(20900 / (10 ** 4 * muZero)) #(19264 / (10 ** 4 * muZero))
exchangeA = 2.625 * 10 ** -7 * 10 ** -4
surface_Ks1 = 3*10**-3 #1*3.49266*10**-3					#changed
surface_Ks2 = 0*10**-3 #1*3.49266*10**-3
surface_Ds1 = 0 * 10 ** -12
surface_Ds2 = 0

#New
#changed
#Aharoni demag factors
NAxx = 0.00063478
NAzz = 0.00260166
NAyy = 1 - NAxx - NAzz

#Bulk easy axis anisotropy
Hubx = 0*40 * 79.57747
Hubz = 0*100 * 79.57747

def radians(x):
	return x * numpy.pi / 180
#changed


#Dependent Variables
if appliedH != 0:
	satMs =  ampMs  * numpy.sign(appliedH)
else:
	satMs = ampMs
Hi = appliedH - NAzz*satMs*numpy.sign(appliedH) + Hubz*numpy.sign(appliedH)	#changed
omegaH =  gamma  *  Hi										#changed  
omegaM =  gamma  *  ampMs  * numpy.sign( appliedH ) 
#Dynamic effective fields due to shape anisotropy and magnetocrystalline anisotropy
omegaUx = gamma * (NAxx * satMs - Hubx)
nAyy = 1 - NAyy
omegaUy = -1 * gamma * satMs * nAyy

alphaExchange = 2 *  exchangeA  / (muZero * ( satMs ** 2)) 
applied_Hu1 = 2 *  surface_Ks1  / (muZero * abs(satMs)) 
applied_Hu2 = 2 *  surface_Ks2  / (muZero * abs(satMs)) 
pinning_d1y = -1 * (2 * surface_Ks1  / (muZero * abs(satMs))) * muZero * abs(satMs) / (2 *  exchangeA )  #changed Readded the factor of 2 in pinning as dispersion code (Mathcad) should now be fixed     
pinning_d2y = -1 * (2 * surface_Ks2  / (muZero * abs(satMs))) * muZero * abs(satMs) / (2 *  exchangeA )  #changed "  ", d1y now depends on Hu1, d2y now on Hu2
pinning_d1x = 0 
pinning_d2x = 0 
bulk_DD1 = 1j *  surface_Ds1  /  exchangeA  
bulk_DD2 = -1j *  surface_Ds2  /  exchangeA 

ind_Variables = {
	"distance_Antennas": distance_Antennas,
	"thicknessSi": thicknessSi,
	"thicknessFM": thicknessFM,
	"centralFreq": centralFreq,
	"appliedH": appliedH,
	"gamma": gamma,
	"ampMs": ampMs,
	"linewidthSlope": linewidthSlope,
	"broadening": broadening,
	"exchangeA": exchangeA,
	"NAxx": NAxx,
	"NAyy": NAyy,
	"NAzz": NAzz,
	"Hubx": Hubx,
	"Hubz": Hubz,
	"surface_Ks1": surface_Ks1,
	"surface_Ks2": surface_Ks2,
	"surface_Ds1": surface_Ds1,
	"surface_Ds2": surface_Ds2
	}

#dep_Variables = {
#	"satMs": ampMs * numpy.sign(appliedH),
#	"omegaH": gamma * appliedH,
#	"omegaM": gamma * ampMs * numpy.sign(appliedH),
#	"alphaExchange": 2 * ind_Variables["exchangeA"] / (muZero * (ampMs * numpy.sign(appliedH)) ** 2),
#	"applied_Hu1": 2 * ind_Variables["surface_Ks1"] / (muZero * abs(ampMs * numpy.sign(appliedH))),
#	"applied_Hu2": 2 * ind_Variables["surface_Ks2"] / (muZero * abs(ampMs * numpy.sign(appliedH))),
#	"pinning_d1y": -1 * (2 * ind_Variables["surface_Ks2"] / (muZero * abs(ampMs * numpy.sign(appliedH)))) * muZero * abs(ampMs * numpy.sign(appliedH)) / (2 * ind_Variables["exchangeA"]),
#	"pinning_d2y": -1 * (2 * ind_Variables["surface_Ks1"] / (muZero * abs(ampMs * numpy.sign(appliedH)))) * muZero * abs(ampMs * numpy.sign(appliedH)) / (2 * ind_Variables["exchangeA"]),
#	"pinning_d1x": 0,
#	"pinning_d2x": 0,
#	"bulk_DD1": 1j * ind_Variables["surface_Ds1"] / ind_Variables["exchangeA"],
#	"bulk_DD2": -1j * ind_Variables["surface_Ds2"] / ind_Variables["exchangeA"]
#	}

def create_global_vars_matrix():
	matrix = numpy.zeros((60,2), dtype = 'U50')
	matrix[0] = ["wsignal:", wsignal]
	matrix[1] = ["wground:", wground]
	matrix[2] = ["wgap:", wgap]
	matrix[3] = ["length_Antenna:", length_Antenna]
	matrix[4] = ["distance_Antennas:", distance_Antennas]
	matrix[5] = ["pts_ground:", pts_ground]
	matrix[6] = ["del_width:", del_width]
	matrix[7] = ["pts_total:", pts_total]
	matrix[8] = ["pts_signal:", pts_signal]
	matrix[9] = ["pts_max:", pts_max]
	matrix[10] = ["epsilonSi:", epsilonSi]
	matrix[11] = ["sigmaFM:", sigmaFM]
	matrix[12] = ["sigmaRu:", sigmaRu]
	matrix[13] = ["sigmaPt:", sigmaPt]
	matrix[14] = ["thicknessSi:", thicknessSi]
	matrix[15] = ["thicknessFM:", thicknessFM]
	matrix[16] = ["thicknessRu:", thicknessRu]
	matrix[17] = ["thicknessPt:", thicknessPt]
	matrix[18] = ["thicknessAl:", thicknessAl]
	matrix[19] = ["resis_Al:", resis_Al]
	matrix[20] = ["var_R1:", var_R1]
	matrix[21] = ["var_R2:", var_R2]
	matrix[22] = ["var_zc:", var_zc]
	matrix[23] = ["freq_lower:", freq_lower/(2*numpy.pi)]				#changed now storing CGS units for Oe and linear frequency
	matrix[24] = ["freq_upper:", freq_upper/(2*numpy.pi)]
	matrix[25] = ["plot_pts_num:", plot_pts_num]
	matrix[26] = ["freq_step:", freq_step/(2*numpy.pi)]
	matrix[27] = ["centralFreq:", centralFreq/(2*numpy.pi)]
	matrix[28] = ["appliedH:", appliedH/79.57747]
	matrix[29] = ["gamma:", gamma/(8*numpy.pi ** 2 * 10 ** 3)]
	matrix[30] = ["ampMs:", ampMs*muZero*10**4]
	matrix[31] = ["linewidthSlope:", linewidthSlope]
	matrix[32] = ["exchangeA:", exchangeA]
	matrix[33] = ["surface_Ks1:", surface_Ks1]
	matrix[34] = ["surface_Ks2:", surface_Ks2]
	matrix[35] = ["surface_Ds1:", surface_Ds1]
	matrix[36] = ["surface_Ds2:", surface_Ds2]
	matrix[37] = ["satMs:", satMs]
	matrix[38] = ["omegaH:", omegaH]
	matrix[39] = ["omegaM:", omegaM]
	matrix[40] = ["alphaExchange:", alphaExchange]
	matrix[41] = ["applied_Hu1:", applied_Hu1]
	matrix[42] = ["applied_Hu2:", applied_Hu2]
	matrix[43] = ["pinning_d1y:", pinning_d1y]
	matrix[44] = ["pinning_d2y:", pinning_d2y]
	matrix[45] = ["pinning_d1x:", pinning_d1x]
	matrix[46] = ["pinning_d2x:", pinning_d2x]
	matrix[47] = ["bulk_DD1:", bulk_DD1]
	matrix[48] = ["bulk_DD2:", bulk_DD2]
	matrix[49] = ["Hubx:", Hubx]
	matrix[50] = ["Hubz:", Hubz]
	matrix[51] = ["Hi:", Hi]
	matrix[52] = ["Hubx: ", Hubx]
	matrix[53] = ["omegaUx: ", omegaUx]
	matrix[54] = ["omegaUy: ", omegaUy]
	matrix[55] = ["var_Ys: ", var_Ys]
	matrix[56] = ["NAxx: ", NAxx]
	matrix[57] = ["NAyy: ", NAyy]
	matrix[58] = ["NAzz: ", NAzz]
	matrix[59] = ["Broadening:", broadening] 
	return matrix

def update_global_vars(indV):
	global distance_Antennas
	global thicknessSi
	global thicknessFM
	global centralFreq
	global appliedH
	global gamma
	global ampMs
	global linewidthSlope
	global exchangeA
	global hub
	global angle
	global surface_Ks1
	global surface_Ks2
	global surface_Ds1
	global surface_Ds2
	global satMs
	global Hubx
	global Hi
	global omegaH
	global omegaM
	global omegaU
	global alphaExchange
	global applied_Hu1
	global applied_Hu2
	global pinning_d1y
	global pinning_d2y
	global pinning_d1x
	global pinning_d2x
	global bulk_DD1
	global bulk_DD2

	#Independent variables
	distance_Antennas = indV["distance_Antennas"]
	thicknessSi = indV["thicknessSi"]
	thicknessFM = indV["thicknessFM"]
	centralFreq = indV["centralFreq"]
	appliedH = indV["appliedH"]
	gamma = indV["gamma"]
	ampMs = indV["ampMs"]
	linewidthSlope = indV["linewidthSlope"]
	exchangeA = indV["exchangeA"]
	surface_Ks1 = indV["surface_Ks1"]
	surface_Ks2 = indV["surface_Ks2"]
	surface_Ds1 = indV["surface_Ds1"]
	surface_Ds2 = indV["surface_Ds2"]

	#Dependent Variables
	if appliedH != 0:
		satMs =  ampMs  * numpy.sign(appliedH)
	else:
		satMs = ampMs
	Hi = appliedH - NAzz*satMs*numpy.sign(appliedH) + Hubz*numpy.sign(appliedH)	#changed
	omegaH= gamma *  Hi
	omegaM= gamma * satMs
	#New effective fields in x and y direction (dynamic components)
	omegaUx = gamma * (NAxx*satMs - Hubx)
	nAyy = 1 - NAyy
	omegaUy = -1*gamma * satMs * nAyy
	alphaExchange= 2 * exchangeA / (muZero * (satMs) ** 2)
	applied_Hu1= 2 * surface_Ks1 / (muZero * abs(satMs))
	applied_Hu2= 2 * surface_Ks2 / (muZero * abs(satMs))
	pinning_d1y= -1 * (applied_Hu1) * muZero * abs(satMs) / (2 * exchangeA)   #Changed No longer dividing Hu1/2, d1y = Hu1
	pinning_d2y= -1 * (applied_Hu2) * muZero * abs(satMs) / (2 * exchangeA)	#Changed No longer dividing by 2, d2y = Hu2
	pinning_d1x= 0
	pinning_d2x= 0
	bulk_DD1= 1j * surface_Ds1 / exchangeA
	bulk_DD2= -1j * surface_Ds2 / exchangeA
	return 0


def update_dict_vars(indV, arg_dict):
	temp = copy.deepcopy(indV)
	temp.update(arg_dict)    #changed
	#for val in args:
	#	if val[0] in temp:
	#		temp[val[0]] = val[1]
	update_global_vars(temp)
	return 0

# Function for complex integration, found at https://stackoverflow.com/questions/5965583/use-scipy-integrate-quad-to-integrate-complex-numbers
def complex_quadrature(func, lower_bound, upper_bound, **kwargs):
    def real_func(*args):
        return numpy.real(func(*args))
    def imag_func(*args):
        return numpy.imag(func(*args))
    real_integral = integrate.quad(real_func, lower_bound, upper_bound, **kwargs)
    imag_integral = integrate.quad(imag_func, lower_bound, upper_bound, **kwargs)
    return real_integral[0] + 1j*imag_integral[0]



def Z0(kk, kks, kkl, kkls):
	numerator = 60 * numpy.pi
	denominator = numpy.sqrt(epsilonEff(kk, kks, kkl, kkls)) * ((kellip(kk) / kellip(kks)) + (kellip(kkl) / kellip(kkls)))
	return (numerator / denominator)
	

def kellip(x):
	#print("x = ", x)
	a = 1
	b = numpy.sqrt(1 - x**2)
	c = x
	K = numpy.pi / (2 * a)
	p = 1
	
	while p >= (10 ** (-10)):
		an = (a + b) / 2
		bn = numpy.sqrt(a*b)
		cn = (a - b) / 2
		#print ("c = ", c)
		#print ("cn = ", cn)
		try:
			p = abs((cn - c) / c)
		except ZeroDivisionError:
			#print("Division by Zero!")
			return K
		#print("p = ", p)
		K = numpy.pi / (2 * a)
		a = an
		b = bn
		c = cn
	return K

def epsilonEff(kk, kks, kkl, kkls):
	numerator = kellip(kks) * kellip(kkl) / (kellip(kk) * kellip(kkls))
	denominator = kellip(kks) * kellip(kkl) / (kellip(kk) * kellip(kkls))
	result = (1 + epsilonSi * numerator) / (1 + denominator)
	return result

@jit(nopython=True)
def gg(x):
	if abs(x) < 40 / thicknessSi:
		return (-1 * numpy.sinh(thicknessSi * abs(x)) / (epsZero * abs(x) * (numpy.sinh(thicknessSi * abs(x)) + epsilonSi * numpy.cosh(thicknessSi * abs(x)))))
	else:
		return (-1 / (epsZero * abs(x) * (1 + epsilonSi))) 

@jit(nopython=True)	
def Ei(x):
	ei = 1
	for n in range(600,0,-1):
		ei = 1 + n / (x + (n + 1) / ei)
	ei = numpy.exp(-x) / (x + 1 / ei)
	return ei

@jit(nopython=True)
def Ci(x):
	result = -1 * (Ei(x * 1j) + Ei(-1j * x)) / 2
	return result
	
@jit(nopython=True)
def create_Gsi_integrand(x,t):
	return numpy.cos(t*x) * gg(x)

@jit(nopython=True)
def gg_eval(k):
	if abs(k) < 40 / thicknessSi:
		return (-1 * numpy.sinh(thicknessSi * abs(k)) / (epsZero * abs(k) * (numpy.sinh(thicknessSi * abs(k)) + epsilonSi * numpy.cosh(thicknessSi * abs(k)))))
	else:
		return (-1 / (epsZero * abs(k) * (1 + epsilonSi))) 

@jit(nopython=True)
def Gsi_integrand(x, t):
	return numpy.cos(t*x) * gg_eval(x)

def Gsi(k):
	#integral = lambda x, t: numpy.cos(t*x) * gg(k)(x)
	#integrated = integrate.quad(integral, 1, 40 / thicknessSi, args =(k), limit = 500)
	integrated = integrate.quad(Gsi_integrand, 1, 40/ thicknessSi, args= (k), limit = 500)
	result = ((integrated[0] + Ci(40 * k / thicknessSi) / (epsZero * (1 + epsilonSi))) / numpy.pi)
	return result.real

def Qp(numsignal, numground, meshpts, nummax, deltaW):
	b = numpy.zeros(meshpts)
	a = numpy.zeros((meshpts, meshpts))
	x = numpy.zeros(meshpts)
	g33 = numpy.zeros(nummax)
	g32 = numpy.zeros(2*nummax)
	g31 = numpy.zeros(2*nummax)
	for i in range(numground):
		b[i] = 0
		b[i+numsignal+numground] = 0
	for i in range(numsignal):
		b[i+numground] = 1
	for i in range(nummax):
		g33[i] = Gsi(i * deltaW)
	for i in range(2*nummax):
		g32[i] = Gsi(i * deltaW + wgap)
		g31[i] = Gsi(i * deltaW + 2 * wgap)
	for i in range(numsignal):
		for j in range(numsignal):
			index = abs(i - j)
			a[i+numground, j+numground] = g33[index] * deltaW
	for i in range(numsignal):
		for j in range(numground):
			indexOne = abs(i+numground-1-j)
			indexTwo = abs(numsignal-1-i+j)
			a[i+numground, j] = g32[indexOne] * deltaW
			a[i+numground, j+numsignal+numground] = g32[indexTwo] * deltaW
	for i in range(numground):
		for j in range(numground):
			index = abs(i - j)
			a[i+numground+numsignal, j+numground+numsignal] = g33[index] * deltaW
			a[i, j] = g33[index] * deltaW
	for i in range(numground):
		for j in range(numsignal):
			indexOne = abs(numground-1-i+j)
			indexTwo = abs(i+numsignal-1-j)
			a[i, j+numground] = g32[indexOne] * deltaW
			a[i+numground+numsignal, j+numground] = g32[indexTwo] * deltaW
	for i in range(numground):
		for j in range(numground):
			indexOne = abs(numground-1-i+j)
			indexTwo = abs(i+numground-1-j)
			a[i,j+numground+numsignal] = g31[indexOne] * deltaW
			a[i+numground+numsignal, j] = g31[indexTwo] * deltaW
	try:
		ww = numpy.linalg.solve(a, b)
	except numpy.linalg.LinAlgError:
		print('LinAlgError. Couldnt solve matrix equation')
		return 1
	jV = ww.sum()

	for i in range(numground):
		b[i] = 1
		b[i+numsignal+numground] = 1
	for i in range(numsignal):
		b[i+numground] = 0
	try:
		ww = numpy.linalg.solve(a, b)
	except:
		print('LinAlgErro. Couldnt solve matrix equation')
		return 1
	jN = ww.sum()
	Cj = -1*jV / jN
	for i in range(numground):
		b[i] = Cj
		b[i+numsignal+numground] = Cj
	for i in range(numsignal):
		b[i+numground] = 1
	ww = numpy.linalg.solve(a, b)
	Q1 = (ww[0:numground].sum() + ww[numground+numsignal:meshpts].sum()) * deltaW
	Css = Q1 / (1 - Cj)
	gw = numpy.zeros(meshpts + 2)
	gw[0] = Css
	gw[1] = Q1
	for i in range(meshpts):
		gw[i+2] = ww[i]
	return gw

def antennaCalcs():
	kk = wsignal / (wsignal + 2 * wgap)
	kks = numpy.sqrt(1 - kk ** 2)
	kkl = numpy.tanh(numpy.pi / 4 *(wsignal / thicknessSi)) / numpy.tanh(numpy.pi / 4 * ((wsignal + 2 * wgap) / thicknessSi))
	kkls = numpy.sqrt(1 - kkl ** 2)
	epsEff = epsilonEff(kk, kks, kkl, kkls)
	z0 = Z0(kk, kks, kkl, kkls)
	gammafs = centralFreq * numpy.sqrt(epsZero * epsEff * muZero) * 1j
	numsignal = 520
	numground = int(numpy.floor(numsignal / 2))
	meshpts = numsignal + 2 * numground
	nummax = max(numsignal, numground)
	deltaW = (wsignal + 2 * wground) / meshpts
	Qpresult = Qp(numsignal, numground, meshpts, nummax, deltaW)
	Ycss = numpy.real(Qpresult[0]) * 1j
	Q1 = (Qpresult[2:numground+2].sum() + Qpresult[numground+numsignal+2:meshpts+2].sum()) * deltaW
	Q2 = (deltaW*Qpresult[numground+2:numground+numsignal+2]).sum()
	return Ycss
	
@jit(nopython=True)
def ww(k, H):
	firstTerm = gamma**2 * H * (H + satMs)
	secondTerm = ((gamma * satMs) ** 2)/4 * (-1 * numpy.exp(-2 * abs(k) * thicknessFM) + 1)
	result = numpy.sqrt(firstTerm + secondTerm)
	return result

def del_H(w):
	result = numpy.sign(appliedH) * 1j * (w * linewidthSlope + broadening / 2)    #Changed now depends on sign of Ms. Also included the inhomogeneous broadening
	return result

@jit(nopython=True)
def Q2(w):
	result = w * sigmaRu * muZero * 1j
	return result

@jit(nopython=True)
def Q4(w):
	result = w * sigmaPt * muZero * 1j
	return result

@jit(nopython=True)
def QQ(w):
	first = abs(numpy.sqrt(Q2(w)))
	second = abs(numpy.sqrt(Q4(w)))
	third = abs(numpy.sqrt(w * sigmaFM * muZero * 1j))
	result = max(first, second, third)
	return result

@jit(nopython=True)
def S0h(K2, k):
	numerator = -1j * k * (K2 * numpy.cosh(K2*thicknessRu) + abs(k) * numpy.sinh(K2*thicknessRu))
	denominator = K2 * (K2 * numpy.sinh(K2 * thicknessRu) + numpy.cosh(K2 * thicknessRu) * abs(k))
	if complex(K2 * thicknessRu).real < 600:
		return numerator / denominator
	else:
		return -k * 1j / K2

@jit(nopython=True)
def SLj(K4, k, w):
	if k < 20 * QQ(w):
		numerator = k * 1j * numpy.exp(-thicknessSi * abs(k))
		denominator = K4 * numpy.sinh(K4 * thicknessPt) + numpy.cosh(K4 * thicknessPt) * abs(k)
		return numerator / denominator
	else:
		result = numpy.sign(k) * numpy.exp(-1 * (thicknessSi+thicknessPt) * abs(k)) * 1j
		return result

@jit(nopython=True)
def SLh(K4, k):
	numerator = k * 1j * (K4 * numpy.cosh(K4 * thicknessPt) + numpy.sinh(K4 * thicknessPt) * abs(k))
	denominator = K4 * (K4 * numpy.sinh(K4 * thicknessPt) + numpy.cosh(K4 * thicknessPt) * abs(k))
	if complex(K4 * thicknessPt).real < 600:
		return numerator / denominator
	else:
		return  k * 1j / K4

@jit(nopython=True)
def Cmy(Q, k, wH, w):
	numerator = -1 * (omegaM * Q * k - w * Q ** 2 + w * k ** 2 + w ** 2 * sigmaFM * muZero * 1j)
	#cmy_denom = numpy.zeros(10, dtype = complex)
	#cmy_denom[0] = w * sigmaFM * wH * muZero
	#cmy_denom[1] = -1 * Q ** 4 * alphaExchange * omegaM * 1j
	#cmy_denom[2] = w * sigmaFM * muZero * omegaM
	#cmy_denom[3] = -1 * Q ** 2 * alphaExchange * w * sigmaFM * muZero * omegaM
	#cmy_denom[4] = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	#cmy_denom[5] = -1 * k ** 2 * wH * 1j
	#cmy_denom[6] = -1 * alphaExchange * k ** 4 * omegaM * 1j
	#cmy_denom[7] = 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j
	#cmy_denom[8] = Q ** 2 * omegaM * 1j
	#cmy_denom[9] = Q ** 2 * wH * 1j

	#cmy_denom_0 = w * sigmaFM * wH * muZero
	#cmy_denom_1 = -1 * Q ** 4 * alphaExchange * omegaM * 1j
	#cmy_denom_2 = w * sigmaFM * muZero * omegaM
	#cmy_denom_3 = -1 * Q ** 2 * alphaExchange * w * sigmaFM * muZero * omegaM
	#cmy_denom_4 = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	#cmy_denom_5 = -1 * k ** 2 * wH * 1j
	#cmy_denom_6 = -1 * alphaExchange * k ** 4 * omegaM * 1j
	#cmy_denom_7 = 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j
	#cmy_denom_8 = Q ** 2 * omegaM * 1j
	#cmy_denom_9 = Q ** 2 * wH * 1j
	#denominator = cmy_denom_0 + cmy_denom_1 + cmy_denom_2 + cmy_denom_3 + cmy_denom_4 + cmy_denom_5 + cmy_denom_6 + cmy_denom_7 + cmy_denom_8 + cmy_denom_9
	denominator = w * sigmaFM * wH * muZero + -1 * Q ** 4 * alphaExchange * omegaM * 1j + w * sigmaFM * muZero * omegaM + -1 * Q ** 2 * alphaExchange * w * sigmaFM * muZero * omegaM + alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM + -1 * k ** 2 * wH * 1j + -1 * alphaExchange * k ** 4 * omegaM * 1j + 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j + Q ** 2 * omegaM * 1j + Q ** 2 * wH * 1j
	#denominator = cmy_denom.sum()
	result = numerator / denominator
	return result

@jit(nopython=True)
def Chx(Q, k, wH, w):
	#num = numpy.zeros(19, dtype = complex)
	#num[0] = Q ** 3 * w * k * 1j
	#num[1] = 2 * Q ** 2 * alphaExchange * k ** 4 * omegaM * 1j
	#num[2] = Q * w ** 2 * sigmaFM * k * muZero
	#num[3] = -1 * Q ** 2 * w * sigmaFM * wH * muZero
	#num[4] = -1 * Q ** 2 * w * sigmaFM * muZero * omegaM
	#num[5] = 2 * w * sigmaFM * k ** 2 * wH * muZero
	#num[6] = w * sigmaFM * k ** 2 * muZero * omegaM
	#num[7] = Q ** 4 * alphaExchange * w * sigmaFM * muZero * omegaM
	#num[8] = 2 * alphaExchange * w * sigmaFM * k ** 4 * muZero * omegaM
	#num[9] =  -3 * Q ** 2 * alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	#num[10] = -1 * Q * w * k ** 3 * 1j
	#num[11] = -1 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	#num[12] = -1 * alphaExchange * k ** 6 * omegaM * 1j
	#num[13] = -1 * k ** 4 * wH * 1j
	#num[14] = -1 * Q ** 4 * alphaExchange * k ** 2 * omegaM * 1j
	#num[15] = w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * 1j
	#num[16] = w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	#num[17] = alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM * 1j
	#num[18] = Q ** 2 * k ** 2 * wH * 1j

	num_0 = Q ** 3 * w * k * 1j
	num_1 = 2 * Q ** 2 * alphaExchange * k ** 4 * omegaM * 1j
	num_2 = Q * w ** 2 * sigmaFM * k * muZero
	num_3 = -1 * Q ** 2 * w * sigmaFM * wH * muZero
	num_4 = -1 * Q ** 2 * w * sigmaFM * muZero * omegaM
	num_5 = 2 * w * sigmaFM * k ** 2 * wH * muZero
	num_6 = w * sigmaFM * k ** 2 * muZero * omegaM
	num_7 = Q ** 4 * alphaExchange * w * sigmaFM * muZero * omegaM
	num_8 = 2 * alphaExchange * w * sigmaFM * k ** 4 * muZero * omegaM
	num_9 =  -3 * Q ** 2 * alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	num_10 = -1 * Q * w * k ** 3 * 1j
	num_11 = -1 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num_12 = -1 * alphaExchange * k ** 6 * omegaM * 1j
	num_13 = -1 * k ** 4 * wH * 1j
	num_14 = -1 * Q ** 4 * alphaExchange * k ** 2 * omegaM * 1j
	num_15 = w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * 1j
	num_16 = w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num_17 = alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM * 1j
	num_18 = Q ** 2 * k ** 2 * wH * 1j

	#numerator = -1 * num.sum()
	numerator = -1 * (num_0 + num_1 + num_2 + num_3 + num_4 + num_5 + num_6 + num_7 + num_8 + num_9 + num_10 + num_11 + num_12 + num_13 + num_14 + num_15 + num_16 + num_17 + num_18)

	denomOne = k ** 2 - Q ** 2 + w * sigmaFM * muZero * 1j

	#denomTwo = numpy.zeros(10, dtype = complex)
	#denomTwo[0] = w * sigmaFM * wH * muZero
	#denomTwo[1] = -1 * Q ** 4 * alphaExchange * omegaM * 1j
	#denomTwo[2] = w * sigmaFM * muZero * omegaM
	#denomTwo[3] = -1 * Q ** 2 * alphaExchange * w * sigmaFM * muZero * omegaM
	#denomTwo[4] = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	#denomTwo[5] = -1 * k ** 2 * wH * 1j
	#denomTwo[6] = -1 * alphaExchange * k ** 4 * omegaM * 1j
	#denomTwo[7] = 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j
	#denomTwo[8] = Q ** 2 * omegaM * 1j
	#denomTwo[9] = Q ** 2 * wH * 1j

	denomTwo_0 = w * sigmaFM * wH * muZero
	denomTwo_1 = -1 * Q ** 4 * alphaExchange * omegaM * 1j
	denomTwo_2 = w * sigmaFM * muZero * omegaM
	denomTwo_3 = -1 * Q ** 2 * alphaExchange * w * sigmaFM * muZero * omegaM
	denomTwo_4 = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM
	denomTwo_5 = -1 * k ** 2 * wH * 1j
	denomTwo_6 = -1 * alphaExchange * k ** 4 * omegaM * 1j
	denomTwo_7 = 2 * Q ** 2 * alphaExchange * k ** 2 * omegaM * 1j
	denomTwo_8 = Q ** 2 * omegaM * 1j
	denomTwo_9 = Q ** 2 * wH * 1j

	#denominator = denomOne * (denomTwo.sum())
	denominator = denomOne * (denomTwo_0 + denomTwo_1 + denomTwo_2 + denomTwo_3 + denomTwo_4 + denomTwo_5 + denomTwo_6 + denomTwo_7 + denomTwo_8 + denomTwo_9)

	return numerator / denominator

@jit(nopython=True)
def Chy(Q, k, wH, w):
	num = numpy.zeros(34, dtype = numpy.complex128)
	num[0] = Q ** 5 * alphaExchange ** 2 * w * sigmaFM * k * muZero * omegaM ** 2
	num[1] = -2 * Q ** 4 * alphaExchange * w * k ** 2 * omegaM * 1j
	num[2] = -1 * Q ** 4 * w * omegaM * 1j
	num[3] = -2 * Q ** 3 * alphaExchange ** 2 * w * sigmaFM * k ** 3 * muZero * omegaM ** 2
	num[4] = -2 * Q ** 3 * alphaExchange * w * sigmaFM * k * wH * muZero * omegaM
	num[5] = -1 * Q ** 3 * alphaExchange * w * sigmaFM * k * muZero * omegaM ** 2
	num[6] = -1 * Q ** 3 * alphaExchange * k ** 3 * omegaM ** 2 * 1j
	num[7] = -1 * Q ** 3 * k * wH * omegaM * 1j
	num[8] = -1 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM * k ** 2 * muZero * omegaM
	num[9] = Q * alphaExchange ** 2 * w * sigmaFM * k ** 5 * muZero * omegaM ** 2
	num[10] = 2 * Q * alphaExchange * w * sigmaFM * k ** 3 * wH * muZero * omegaM
	num[11] = Q * alphaExchange * w * sigmaFM * k ** 3 * muZero * omegaM ** 2
	num[12] = Q * w * sigmaFM * k * wH ** 2 * muZero 
	num[13] = Q * w * sigmaFM * k * wH * muZero * omegaM
	num[14] = -1 * alphaExchange * w ** 3 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM * 1j
	num[15] = alphaExchange * w ** 2 * sigmaFM * k ** 4 * muZero * omegaM 
	num[16] = -1 * w ** 3 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num[17] = w ** 2 * sigmaFM * k ** 2 * wH * muZero
	num[18] = -1 * Q ** 4 * w * wH * 1j
	num[19] = -4 * Q ** 3 * alphaExchange * k ** 3 * wH * omegaM * 1j
	num[20] = Q * alphaExchange ** 2 * k ** 7 * omegaM ** 2 * 1j
	num[21] = -1 * Q ** 7 * alphaExchange ** 2 * k * omegaM ** 2 * 1j
	num[22] = -3 * Q ** 3 * alphaExchange ** 2 * k ** 5 * omegaM ** 2 * 1j
	num[23] = -1 * w ** 3 * sigmaFM ** 2 * wH * muZero ** 2 * 1j
	num[24] = -1 * Q ** 3 * k * wH ** 2 * 1j
	num[25] = 2 * Q * alphaExchange * k ** 5 * wH * omegaM * 1j
	num[26] = 2 * Q ** 5 * alphaExchange * k * wH * omegaM * 1j
	num[27] = 3 * Q ** 5 * alphaExchange ** 2 * k ** 3 * omegaM ** 2 * 1j
	num[28] = Q * k ** 3 * wH ** 2 * 1j
	num[29] = Q ** 5 * alphaExchange * k * omegaM ** 2 * 1j
	num[30] = Q ** 2 * w * k ** 2 * wH * 1j
	num[31] = Q ** 2 * alphaExchange * w * k ** 4 * omegaM * 1j
	num[32] = Q ** 2 * alphaExchange * w ** 3 * sigmaFM ** 2 * muZero ** 2 * omegaM * 1j
	num[33] = Q ** 6 * alphaExchange * w * omegaM * 1j
	
	numerator = -1 * num.sum()

	denom = numpy.zeros(28)
	denom[0] = Q ** 8 * alphaExchange ** 2 * omegaM ** 2
	denom[1] = -4 * Q ** 6 * alphaExchange ** 2 * k ** 2 * omegaM ** 2
	denom[2] = -2 * Q ** 6 * alphaExchange * wH * omegaM
	denom[3] = -2 * Q ** 6 * alphaExchange * omegaM ** 2
	denom[4] = Q ** 4 * alphaExchange ** 2 * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM ** 2
	denom[5] = 6 * Q ** 4 * alphaExchange ** 2 * k ** 4 * omegaM ** 2
	denom[6] = 6 * Q ** 4 * alphaExchange * k ** 2 * wH * omegaM 
	denom[7] = 4 * Q ** 4 * alphaExchange * k ** 2 * omegaM ** 2
	denom[8] = Q ** 4 * wH ** 2
	denom[9] = 2 * Q ** 4 * wH * omegaM 
	denom[10] = Q ** 4 * omegaM ** 2
	denom[11] = -2 * Q ** 2 * alphaExchange ** 2 * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM ** 2
	denom[12] = -4 * Q ** 2 * alphaExchange ** 2 * k ** 6 * omegaM ** 2
	denom[13] = -2 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * omegaM
	denom[14] = -2 * Q ** 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM ** 2
	denom[15] = -6 * Q ** 2 * alphaExchange * k ** 4 * wH * omegaM
	denom[16] = -2 * Q ** 2 * alphaExchange * k ** 4 * omegaM ** 2
	denom[17] = -2 * Q ** 2 * k ** 2 * wH ** 2 
	denom[18] = 2 * Q ** 2 * k ** 2 * wH * omegaM
	denom[19] = alphaExchange ** 2 * w ** 2 * sigmaFM ** 2 * k ** 4 * muZero ** 2 * omegaM ** 2
	denom[20] = alphaExchange ** 2 * k ** 8 * omegaM ** 2
	denom[21] = 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * wH * muZero ** 2 * omegaM
	denom[22] = 2 * alphaExchange * w ** 2 * sigmaFM ** 2 * k ** 2 * muZero ** 2 * omegaM ** 2
	denom[23] = 2 * alphaExchange * k ** 6 * wH * omegaM
	denom[24] = w ** 2 * sigmaFM ** 2 * wH ** 2 * muZero ** 2
	denom[25] = 2 * w ** 2 * sigmaFM ** 2 * wH * muZero ** 2 * omegaM
	denom[26] = w ** 2 * sigmaFM ** 2 * muZero ** 2 * omegaM ** 2
	denom[27] = k ** 4 * wH ** 2
	denominator = denom.sum()

	return numerator / denominator

@jit(nopython=True)
def create_b_var(k, wH, w):
	firstTerm = -3 * alphaExchange ** 2 * k ** 2 * omegaM ** 2
	secondTerm = -alphaExchange * omegaM ** 2
	thirdTerm = -2 * wH * alphaExchange * omegaM
	fourthTerm = -1 * alphaExchange ** 2 * w * sigmaFM * muZero * omegaM ** 2 * 1j
	fifthTerm = -1 * alphaExchange * omegaM * (omegaUx + omegaUy)			#changed
	result = firstTerm + secondTerm + thirdTerm + fourthTerm + fifthTerm
	return result

@jit(nopython=True)
def create_c_var(k, wH, w):
	firstTerm = numpy.zeros(2, dtype = numpy.complex128)
	firstTerm[0] = 3 * k ** 4 * omegaM ** 2
	firstTerm[1] = 2 * w * sigmaFM * k ** 2 * muZero * omegaM ** 2 * 1j

	secondTerm = numpy.zeros(4, dtype = numpy.complex128)
	secondTerm[0] = 2 * k ** 2 * omegaM ** 2
	secondTerm[1] = 4 * wH * k ** 2 * omegaM
	secondTerm[2] = 2 * w * sigmaFM * muZero * omegaM ** 2 * 1j
	secondTerm[3] = 2 * w * sigmaFM * wH * muZero * omegaM * 1j

	thirdTerm = wH ** 2
	fourthTerm = -1 * w ** 2
	fifthTerm = omegaM * wH
	#changed
	sixthTerm = -1 * ((-2 * alphaExchange * omegaM * k ** 2 - wH - omegaUx - alphaExchange * w * sigmaFM * muZero * omegaM * 1j) * omegaUy - omegaUx * (2 * alphaExchange * omegaM * k ** 2 + wH + omegaM + alphaExchange * w * sigmaFM * muZero * omegaM * 1j))

	result = firstTerm.sum() * alphaExchange ** 2 + secondTerm.sum() * alphaExchange + thirdTerm + fourthTerm + fifthTerm + sixthTerm
	return result

@jit(nopython=True)
def create_d_var(k, wH, w):
	firstTerm = numpy.zeros(2, dtype = numpy.complex128)
	firstTerm[0] = -1 * k ** 6 * omegaM ** 2
	firstTerm[1] = -1 * w * sigmaFM * k ** 4 * muZero * omegaM ** 2 * 1j

	secondTerm = numpy.zeros(4, dtype = numpy.complex128)
	secondTerm[0] = -1 * k ** 4 * omegaM ** 2
	secondTerm[1] = -2 * wH * k ** 4 * omegaM
	secondTerm[2] = -2 * w * sigmaFM * wH * muZero * k ** 2 * omegaM * 1j
	secondTerm[3] = -2 * w * sigmaFM * k ** 2 * muZero * omegaM ** 2 * 1j

	thirdTerm = numpy.zeros(7, dtype = numpy.complex128)
	thirdTerm[0] = w ** 2 * k ** 2
	thirdTerm[1] = -2 * sigmaFM * muZero * w * wH * omegaM * 1j
	thirdTerm[2] = -1 * k ** 2 * wH ** 2
	thirdTerm[3] = -1 * k ** 2 * wH * omegaM
	thirdTerm[4] = -1 * w * sigmaFM * wH ** 2 * muZero * 1j
	thirdTerm[5] = -1 * w * sigmaFM * muZero * omegaM ** 2 * 1j
	thirdTerm[6] = w ** 3 * sigmaFM * muZero * 1j

	#changed
	fourthTerm = numpy.zeros(6, dtype = numpy.complex128)
	fourthTerm[0] = k ** 2 * wH
	fourthTerm[1] = omegaUy * (k ** 2 + w * sigmaFM * muZero * 1j)
	fourthTerm[2] = alphaExchange * k ** 4 * omegaM
	fourthTerm[3] = w * sigmaFM * muZero * omegaM * 1j
	fourthTerm[4] = w * sigmaFM * muZero * wH * 1j
	fourthTerm[5] = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM * 1j
	
	fifthTerm = numpy.zeros(6, dtype = numpy.complex128)
	fifthTerm[0] = k ** 2 * wH
	fifthTerm[1] = k ** 2 * omegaM
	fifthTerm[2] = alphaExchange * k ** 4 * omegaM
	fifthTerm[3] = w * sigmaFM * muZero * omegaM * 1j
	fifthTerm[4] = w * sigmaFM * muZero * wH * 1j
	fifthTerm[5] = alphaExchange * w * sigmaFM * k ** 2 * muZero * omegaM * 1j
	#changed
	result = firstTerm.sum() * alphaExchange ** 2 + secondTerm.sum() * alphaExchange + thirdTerm.sum() - fourthTerm.sum() * omegaUx - fifthTerm.sum() * omegaUy
	return result

@jit(nopython=True)
def create_DD_var(a, b, c, d):
	term = numpy.zeros(8, dtype = numpy.complex128)
	term[0] = c ** 3 / (27 * a ** 3)
	term[1] = d ** 2 / (4 * a ** 2)
	term[2] = (b ** 3 * d) / (27 * a ** 4)
	term[3] = -1 * (b ** 2 * c ** 2) / (108 * a ** 4)
	term[4] = -1 * (b * c * d) / (6 * a ** 3)
	term[5] = -1 * b ** 3 / (27 * a ** 3)
	term[6] = -1 * d / (2 * a)
	term[7] = b * c / (6 * a ** 2)

	root = numpy.sqrt(term[0:5].sum())
	nonroot = term[5:].sum()
	result = (root + nonroot) ** (1/3)
	return result

@jit(nopython=True)
def create_Q1_var(a, b, c, DD):
	term = numpy.zeros(3, dtype = numpy.complex128)
	term[0] = DD
	term[1] = (c / (3*a) - b ** 2 / (9 * a ** 2)) / DD
	term[2] = b/ (3 * a)
	result = numpy.sqrt(term[0] - term[1] - term[2])
	return result

@jit(nopython=True)
def create_Q2_var(a, b, c, DD):
	term = numpy.zeros(3, dtype = numpy.complex128)
	term[0] = c / (3 * a)
	term[1] = -1 * b ** 2 / (9 * a ** 2)
	term[2] = -1 * b / (3 * a)
	realTerm = (term[0] + term[1]) / (2 * DD) + term[2] - DD / 2
	imagTerm = -1j * (numpy.sqrt(3) * ((term[0] + term[1]) / DD + DD)) / 2
	result = numpy.sqrt(realTerm + imagTerm)
	return result

@jit(nopython=True)
def create_Q3_var(a, b, c, DD):
	term = numpy.zeros(6, dtype = numpy.complex128)
	term[0] = c / (3 * a)
	term[1] = -1 * b ** 2 / (9 * a ** 2)
	term[2] = -1 * b / (3 * a)
	term[3] = -1 * DD / 2
	term[4] = ((1 / 3) / a) * c
	term[5] = -1 * ((1 / 9) / a ** 2) * b ** 2
	realTerm = (term[0] + term[1]) / (2 * DD) + term[2] + term[3]
	imagTerm = (1 / 2) * numpy.sqrt(3) * ((term[4] + term[5]) / DD + DD) * 1j
	result = numpy.sqrt(realTerm + imagTerm)
	return result



@jit(nopython=True)
def create_Cmy_vec(Q1, Q2, Q3, k, wH, w):
	temp_Cmy = numpy.zeros(6, dtype = numpy.complex128)
	temp_Cmy[0] = Cmy(Q1, k, wH, w)
	temp_Cmy[1] = Cmy(Q2, k, wH, w)
	temp_Cmy[2] = Cmy(Q3, k, wH, w)
	temp_Cmy[3] = Cmy(-Q1, k, wH, w)
	temp_Cmy[4] = Cmy(-Q2, k, wH, w)
	temp_Cmy[5] = Cmy(-Q3, k, wH, w)

	#temp_Cmy_0 = Cmy(Q1, k, wH, w)
	#temp_Cmy_1 = Cmy(Q2, k, wH, w)
	#temp_Cmy_2 = Cmy(Q3, k, wH, w)
	#temp_Cmy_3 = Cmy(-Q1, k, wH, w)
	#temp_Cmy_4 = Cmy(-Q2, k, wH, w)
	#temp_Cmy_5 = Cmy(-Q3, k, wH, w)
	return temp_Cmy


@jit(nopython=True)
def create_Chx_vec(Q1, Q2, Q3, k, wH, w):
	temp_Chx = numpy.zeros(6, dtype = numpy.complex128)
	temp_Chx[0] = Chx(Q1, k, wH, w)
	temp_Chx[1] = Chx(Q2, k, wH, w)
	temp_Chx[2] = Chx(Q3, k, wH, w)
	temp_Chx[3] = Chx(-Q1, k, wH, w)
	temp_Chx[4] = Chx(-Q2, k, wH, w)
	temp_Chx[5] = Chx(-Q3, k, wH, w)

	#temp_Chx_0 = Chx(Q1, k, wH, w)
	#temp_Chx_1 = Chx(Q2, k, wH, w)
	#temp_Chx_2 = Chx(Q3, k, wH, w)
	#temp_Chx_3 = Chx(-Q1, k, wH, w)
	#temp_Chx_4 = Chx(-Q2, k, wH, w)
	#temp_Chx_5 = Chx(-Q3, k, wH, w)
	#return (temp_Chx_0, temp_Chx_1, temp_Chx_2, temp_Chx_3, temp_Chx_4, temp_Chx_5)
	return temp_Chx

@jit(nopython = True)
def create_A_matrix(Q1, Q2, Q3, k, Cmy_vec, Chx_vec, SS0h, SSLh, Rm, Rh):
	temp_A = numpy.zeros((6,6), dtype = numpy.complex128)
	temp_A[0,0] = Q1 - pinning_d1x + bulk_DD1 * k * Cmy_vec[0]
	temp_A[0,1] = Q2 - pinning_d1x + bulk_DD1 * k * Cmy_vec[1]
	temp_A[0,2] = Q3 - pinning_d1x + bulk_DD1 * k * Cmy_vec[2]
	temp_A[0,3] = -Q1 - pinning_d1x + bulk_DD1 * k * Cmy_vec[3]
	temp_A[0,4] = -Q2 - pinning_d1x + bulk_DD1 * k * Cmy_vec[4]
	temp_A[0,5] = -Q3 - pinning_d1x + bulk_DD1 * k * Cmy_vec[5]
	temp_A[1,0] = (Q1 + pinning_d2x - bulk_DD2 * k * Cmy_vec[0]) * numpy.exp(Q1 * thicknessFM)
	temp_A[1,1] = (Q2 + pinning_d2x - bulk_DD2 * k * Cmy_vec[1]) * numpy.exp(Q2 * thicknessFM)
	temp_A[1,2] = (Q3 + pinning_d2x - bulk_DD2 * k * Cmy_vec[2]) * numpy.exp(Q3 * thicknessFM)
	temp_A[1,3] = (-Q1 + pinning_d2x - bulk_DD2 * k * Cmy_vec[3]) * numpy.exp(-Q1 * thicknessFM)
	temp_A[1,4] = (-Q2 + pinning_d2x - bulk_DD2 * k * Cmy_vec[4]) * numpy.exp(-Q2 * thicknessFM)
	temp_A[1,5] = (-Q3 + pinning_d2x - bulk_DD2 * k * Cmy_vec[5]) * numpy.exp(-Q3 * thicknessFM)
	temp_A[2,0] = Cmy_vec[0] * (Q1 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,1] = Cmy_vec[1] * (Q2 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,2] = Cmy_vec[2] * (Q3 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,3] = Cmy_vec[3] * (-Q1 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,4] = Cmy_vec[4] * (-Q2 - pinning_d1y) - bulk_DD1 * k
	temp_A[2,5] = Cmy_vec[5] * (-Q3 - pinning_d1y) - bulk_DD1 * k
	temp_A[3,0] = (Cmy_vec[0] * (Q1 + pinning_d2y) + bulk_DD2 * k) * numpy.exp(Q1 * thicknessFM)
	temp_A[3,1] = (Cmy_vec[1] * (Q2 + pinning_d2y) + bulk_DD2 * k) * numpy.exp(Q2 * thicknessFM)
	temp_A[3,2] = (Cmy_vec[2] * (Q3 + pinning_d2y) + bulk_DD2 * k) * numpy.exp(Q3 * thicknessFM)
	temp_A[3,3] = (Cmy_vec[3] * (-Q1 + pinning_d2y) + bulk_DD2 * k) * numpy.exp(-Q1 * thicknessFM)
	temp_A[3,4] = (Cmy_vec[4] * (-Q2 + pinning_d2y) + bulk_DD2 * k) * numpy.exp(-Q2 * thicknessFM)
	temp_A[3,5] = (Cmy_vec[5] * (-Q3 + pinning_d2y) + bulk_DD2 * k) * numpy.exp(-Q3 * thicknessFM)
	temp_A[4,0] = Cmy_vec[0] * Rm + Chx_vec[0] * (SS0h + Rh * Q1)
	temp_A[4,1] = Cmy_vec[1] * Rm + Chx_vec[1] * (SS0h + Rh * Q2)
	temp_A[4,2] = Cmy_vec[2] * Rm + Chx_vec[2] * (SS0h + Rh * Q3)
	temp_A[4,3] = Cmy_vec[3] * Rm + Chx_vec[3] * (SS0h + Rh * (-Q1))
	temp_A[4,4] = Cmy_vec[4] * Rm + Chx_vec[4] * (SS0h + Rh * (-Q2))
	temp_A[4,5] = Cmy_vec[5] * Rm + Chx_vec[5] * (SS0h + Rh * (-Q3))
	temp_A[5,0] = (Cmy_vec[0] * Rm + Chx_vec[0] * (SSLh + Rh * Q1)) * numpy.exp(Q1 * thicknessFM)
	temp_A[5,1] = (Cmy_vec[1] * Rm + Chx_vec[1] * (SSLh + Rh * Q2)) * numpy.exp(Q2 * thicknessFM)
	temp_A[5,2] = (Cmy_vec[2] * Rm + Chx_vec[2] * (SSLh + Rh * Q3)) * numpy.exp(Q3 * thicknessFM)
	temp_A[5,3] = (Cmy_vec[3] * Rm + Chx_vec[3] * (SSLh + Rh * (-Q1))) * numpy.exp(-Q1 * thicknessFM)
	temp_A[5,4] = (Cmy_vec[4] * Rm + Chx_vec[4] * (SSLh + Rh * (-Q2))) * numpy.exp(-Q2 * thicknessFM)
	temp_A[5,5] = (Cmy_vec[5] * Rm + Chx_vec[5] * (SSLh + Rh * (-Q3))) * numpy.exp(-Q3 * thicknessFM)

	return temp_A

@jit(nopython=True)
def create_B_matrix(A):
	temp_B = numpy.zeros((4,4), dtype = numpy.complex128)
	temp_B[0,0] = A[0,0]*A[1,4]*A[2,5] - A[0,0]*A[1,5]*A[2,4] - A[1,0]*A[0,4]*A[2,5] + A[1,0]*A[0,5]*A[2,4] + A[2,0]*A[0,4]*A[1,5] - A[2,0]*A[0,5]*A[1,4]
	temp_B[0,1] = A[0,1]*A[1,4]*A[2,5] - A[0,1]*A[1,5]*A[2,4] - A[1,1]*A[0,4]*A[2,5] + A[1,1]*A[0,5]*A[2,4] + A[2,1]*A[0,4]*A[1,5] - A[2,1]*A[0,5]*A[1,4]
	temp_B[0,2] = A[0,2]*A[1,4]*A[2,5] - A[0,2]*A[1,5]*A[2,4] - A[1,2]*A[0,4]*A[2,5] + A[1,2]*A[0,5]*A[2,4] + A[2,2]*A[0,4]*A[1,5] - A[2,2]*A[0,5]*A[1,4]
	temp_B[0,3] = A[0,3]*A[1,4]*A[2,5] - A[0,3]*A[1,5]*A[2,4] - A[1,3]*A[0,4]*A[2,5] + A[1,3]*A[0,5]*A[2,4] + A[2,3]*A[0,4]*A[1,5] - A[2,3]*A[0,5]*A[1,4]
	temp_B[1,0] = A[0,0]*A[1,4]*A[3,5] - A[0,0]*A[1,5]*A[3,4] - A[1,0]*A[0,4]*A[3,5] + A[1,0]*A[0,5]*A[3,4] + A[3,0]*A[0,4]*A[1,5] - A[3,0]*A[0,5]*A[1,4]
	temp_B[1,1] = A[0,1]*A[1,4]*A[3,5] - A[0,1]*A[1,5]*A[3,4] - A[1,1]*A[0,4]*A[3,5] + A[1,1]*A[0,5]*A[3,4] + A[3,1]*A[0,4]*A[1,5] - A[3,1]*A[0,5]*A[1,4]
	temp_B[1,2] = A[0,2]*A[1,4]*A[3,5] - A[0,2]*A[1,5]*A[3,4] - A[1,2]*A[0,4]*A[3,5] + A[1,2]*A[0,5]*A[3,4] + A[3,2]*A[0,4]*A[1,5] - A[3,2]*A[0,5]*A[1,4]
	temp_B[1,3] = A[0,3]*A[1,4]*A[3,5] - A[0,3]*A[1,5]*A[3,4] - A[1,3]*A[0,4]*A[3,5] + A[1,3]*A[0,5]*A[3,4] + A[3,3]*A[0,4]*A[1,5] - A[3,3]*A[0,5]*A[1,4]
	temp_B[2,0] = A[0,0]*A[1,4]*A[4,5] - A[0,0]*A[1,5]*A[4,4] - A[1,0]*A[0,4]*A[4,5] + A[1,0]*A[0,5]*A[4,4] + A[4,0]*A[0,4]*A[1,5] - A[4,0]*A[0,5]*A[1,4]
	temp_B[2,1] = A[0,1]*A[1,4]*A[4,5] - A[0,1]*A[1,5]*A[4,4] - A[1,1]*A[0,4]*A[4,5] + A[1,1]*A[0,5]*A[4,4] + A[4,1]*A[0,4]*A[1,5] - A[4,1]*A[0,5]*A[1,4]
	temp_B[2,2] = A[0,2]*A[1,4]*A[4,5] - A[0,2]*A[1,5]*A[4,4] - A[1,2]*A[0,4]*A[4,5] + A[1,2]*A[0,5]*A[4,4] + A[4,2]*A[0,4]*A[1,5] - A[4,2]*A[0,5]*A[1,4]
	temp_B[2,3] = A[0,3]*A[1,4]*A[4,5] - A[0,3]*A[1,5]*A[4,4] - A[1,3]*A[0,4]*A[4,5] + A[1,3]*A[0,5]*A[4,4] + A[4,3]*A[0,4]*A[1,5] - A[4,3]*A[0,5]*A[1,4]
	temp_B[3,0] = A[0,0]*A[1,4]*A[5,5] - A[0,0]*A[1,5]*A[5,4] - A[1,0]*A[0,4]*A[5,5] + A[1,0]*A[0,5]*A[5,4] + A[5,0]*A[0,4]*A[1,5] - A[5,0]*A[0,5]*A[1,4]
	temp_B[3,1] = A[0,1]*A[1,4]*A[5,5] - A[0,1]*A[1,5]*A[5,4] - A[1,1]*A[0,4]*A[5,5] + A[1,1]*A[0,5]*A[5,4] + A[5,1]*A[0,4]*A[1,5] - A[5,1]*A[0,5]*A[1,4]
	temp_B[3,2] = A[0,2]*A[1,4]*A[5,5] - A[0,2]*A[1,5]*A[5,4] - A[1,2]*A[0,4]*A[5,5] + A[1,2]*A[0,5]*A[5,4] + A[5,2]*A[0,4]*A[1,5] - A[5,2]*A[0,5]*A[1,4]
	temp_B[3,3] = A[0,3]*A[1,4]*A[5,5] - A[0,3]*A[1,5]*A[5,4] - A[1,3]*A[0,4]*A[5,5] + A[1,3]*A[0,5]*A[5,4] + A[5,3]*A[0,4]*A[1,5] - A[5,3]*A[0,5]*A[1,4]

	return temp_B

@jit(nopython=True)
def create_M_vec(A, B, DD, Det):
	M = numpy.zeros(6, dtype = numpy.complex128)
	M[0] = -(DD / Det) * (B[0,1]*B[1,2]*B[2,3] - B[0,1]*B[1,3]*B[2,2] - B[0,2]*B[1,1]*B[2,3] + B[0,2]*B[2,1]*B[1,3] + B[1,1]*B[0,3]*B[2,2] - B[0,3]*B[1,2]*B[2,1])
	M[1] = (DD / Det) * (B[0,0]*B[1,2]*B[2,3] - B[0,0]*B[1,3]*B[2,2] - B[1,0]*B[0,2]*B[2,3] + B[1,0]*B[0,3]*B[2,2] + B[0,2]*B[2,0]*B[1,3] - B[2,0]*B[0,3]*B[1,2])
	M[2] = -(DD / Det) * (B[0,0]*B[1,1]*B[2,3] - B[0,0]*B[2,1]*B[1,3] - B[0,1]*B[1,0]*B[2,3] + B[0,1]*B[2,0]*B[1,3] + B[1,0]*B[0,3]*B[2,1] - B[1,1]*B[2,0]*B[0,3])
	M[3] = (DD / Det) * (B[0,0]*B[1,1]*B[2,2] - B[0,0]*B[1,2]*B[2,1] - B[0,1]*B[1,0]*B[2,2] + B[0,1]*B[2,0]*B[1,2] + B[1,0]*B[0,2]*B[2,1] - B[0,2]*B[1,1]*B[2,0])
	numerator = -1 * (A[0,0]*A[1,5]*M[0] - A[1,0]*A[0,5]*M[0] + A[0,1]*A[1,5]*M[1] - A[1,1]*A[0,5]*M[1] + A[0,2]*A[1,5]*M[2] - A[1,2]*A[0,5]*M[2] + A[0,3]*A[1,5]*M[3] - A[1,3]*A[0,5]*M[3])
	denominator = A[0,4]*A[1,5] - A[0,5]*A[1,4]
	M[4] = numerator / denominator
	M[5] = -1 * (A[0,0]*M[0] + A[0,1]*M[1] + A[0,2]*M[2] + A[0,3]*M[3] + A[0,4]*M[4]) / A[0,5]
	return M

@jit(nopython=True)
def create_hxk_var(Chx_vec, M_vec, Q1, Q2, Q3):
	term = numpy.zeros(6, dtype = numpy.complex128)
	term[0] = Chx_vec[0] * M_vec[0] * numpy.exp(Q1 * thicknessFM)
	term[1] = Chx_vec[1] * M_vec[1] * numpy.exp(Q2 * thicknessFM)
	term[2] = Chx_vec[2] * M_vec[2] * numpy.exp(Q3 * thicknessFM)
	term[3] = Chx_vec[3] * M_vec[3] * numpy.exp(-Q1 * thicknessFM)
	term[4] = Chx_vec[4] * M_vec[4] * numpy.exp(-Q2 * thicknessFM)
	term[5] = Chx_vec[5] * M_vec[5] * numpy.exp(-Q3 * thicknessFM)
	result = term.sum()
	return result

@jit(nopython=True)
def create_hyl_var(k, K4, hxk):
	numerator = -1 * numpy.sign(k) * 1j * k ** 2 * hxk * numpy.exp(-1 * abs(k) * thicknessSi)
	denominator = k ** 2 * numpy.cosh(K4 * thicknessPt) + K4 * numpy.sinh(K4 * thicknessPt) * abs(k)
	result = numerator / denominator
	return result

@jit(nopython=True)
def MM(k, wH, w):
	var_K2 = numpy.sqrt(Q2(w) + k ** 2)
	#print("K2: ", var_K2)
	var_K4 = numpy.sqrt(Q4(w) + k ** 2)
	#print("K4: ", var_K4)
	Rm = (k ** 4 - w * sigmaFM * k ** 2 * muZero * 1j) / (w ** 2 * sigmaFM ** 2 * muZero ** 2 + k ** 4)
	#print("Rm: ", Rm)
	SS0h = S0h(var_K2, k)
	#print("SS0h: ", SS0h)
	SSLh = SLh(var_K4, k)
	#print("SSLh: ", SSLh)
	Rh = (w * sigmaFM * k * muZero + k ** 3 * 1j) / (w ** 2 * sigmaFM ** 2 * muZero ** 2 + k ** 4)
	#print("Rh: ", Rh)
	SSLj = SLj(var_K4, k, w)
	#print("SSLj: ", SSLj)
	var_b = create_b_var(k, wH, w)
	#print("b: ", var_b)
	var_a = alphaExchange ** 2 * omegaM ** 2
	#print("a: ", var_a)
	var_c = create_c_var(k, wH, w)
	#print("c: ", var_c)
	var_d = create_d_var(k, wH, w)
	#print("d: ", var_d)
	var_DD = create_DD_var(var_a, var_b, var_c, var_d)
	#print("DD: ", var_DD)
	var_Q1 = create_Q1_var(var_a, var_b, var_c, var_DD)
	#print("Q1: ", var_Q1)
	var_Q2 = create_Q2_var(var_a, var_b, var_c, var_DD)
	#print("Q2: ", var_Q2)
	var_Q3 = create_Q3_var(var_a, var_b, var_c, var_DD)
	#print("Q3: ", var_Q3)
	vec_Cmy = create_Cmy_vec(var_Q1, var_Q2, var_Q3, k, wH, w)
	#print("Cmy: ", vec_Cmy)
	vec_Chx = create_Chx_vec(var_Q1, var_Q2, var_Q3, k, wH, w)
	#print("Chx: ", vec_Chx)
	matrix_A = create_A_matrix(var_Q1, var_Q2, var_Q3, k, vec_Cmy, vec_Chx, SS0h, SSLh, Rm, Rh)
	matrix_B = create_B_matrix(matrix_A)
	det_B = numpy.linalg.det(matrix_B)
	var_F = -SSLj
	#print("F: ", var_F)
	var_DD = var_F * (matrix_A[0,4] * matrix_A[1,5] - matrix_A[0,5] * matrix_A[1,4])
	#print("DD2: ", var_DD)
	vec_M = create_M_vec(matrix_A, matrix_B, var_DD, det_B)
	var_hxk = create_hxk_var(vec_Chx, vec_M, var_Q1, var_Q2, var_Q3)
	#print("hxk: ", var_hxk)
	var_hyl = create_hyl_var(k, var_K4, var_hxk)
	#print("hyl: ", var_hyl)
	result_ek = -1 * (w * muZero * var_hyl) / k
	#print("ek: ", result_ek)
	return result_ek

@jit(nopython=True)
def create_Gind_integral(k, z, w):
	arg_One = numpy.sqrt(Q4(w) + k ** 2)
	arg_Two = thicknessSi * abs(k)

	num = numpy.zeros(2, dtype = numpy.complex128)
	num[0] = k ** 2 * numpy.cosh(arg_One * thicknessPt) * numpy.cosh(arg_Two)
	num[1] = arg_One * numpy.sinh(arg_One * thicknessPt) * numpy.sinh(arg_Two) * abs(k)
	numerator = -1j * num.sum() * numpy.exp(-abs(k) * thicknessSi) * numpy.cos(k * z)

	denom = numpy.zeros(2, dtype = numpy.complex128)
	denom[0] = k ** 2 * numpy.cosh(arg_One * thicknessPt)
	denom[1] = arg_One * numpy.sinh(arg_One * thicknessPt) * abs(k)
	denominator = abs(k) * denom.sum()
	result = numerator / denominator
	return result

@jit(nopython=True)
def realFunc(x, z, w):
	return numpy.real(create_Gind_integral(x, z, w))
@jit(nopython=True)
def imagFunc(x, z, w):
	return numpy.imag(create_Gind_integral(x, z, w))
	
def Gind(z, w):
	upper_Bound = 20 * QQ(w)		#changed upperbound to 10*QQ from previous 20*QQ
	lower_Bound = 0
	integral = complex_quadrature(create_Gind_integral, lower_Bound, upper_Bound, args=(z,w), limit=1000)

	#int_real = integrate.quad(realFunc, lower_Bound, upper_Bound, args=(z,w), limit=1000)[0]
	#int_imag = integrate.quad(imagFunc, lower_Bound, upper_Bound, args=(z,w), limit=1000)[0]
	#integral = int_real + int_imag * 1j
	first_Term = 2 * (-w * muZero) / (2 * numpy.pi) * integral
	argument_One = 2 * (thicknessSi + thicknessPt) * upper_Bound - upper_Bound * z * 1j
	argument_Two = 2 * (thicknessSi + thicknessPt) * upper_Bound + upper_Bound * z * 1j
	#second_Term = 2 * (-w * muZero * -1j) / (2 * numpy.pi) * ((Ei(argument_One) + Ei(argument_Two)) / 2 - Ci(z * upper_Bound))      #changed this is the old term where we still have Ei's and do not divide Ci by 2
	second_Term = 2 * (-w * muZero * -1j) / (2 * numpy.pi) * (- Ci(z * upper_Bound)/2)   #changed new term for Gind, removed the Ei terms and are now dividing Ci by 2
	result = first_Term + second_Term
	#return 0					# changed making Gind 0 to remove inductive coupling
	return result
	
@jit(nopython=True)
def create_eG_integral(k, H, z, w):
	first_Term = MM(k, H * gamma, w) * numpy.exp(-1j * k * z)
	second_Term = MM(-k, H * gamma, w) * numpy.exp(1j * k * z)
	result = first_Term + second_Term
	return result

@jit(nopython=True)
def real_func(k, H, z, w):
	return numpy.real(create_eG_integral(k, H, z, w))
@jit(nopython=True)
def imag_func(k, H, z, w):
		return numpy.imag(create_eG_integral(k, H, z, w))

def eG(H, z, w):
	upper_Bound = 100 * numpy.pi / (wsignal)
	lower_Bound = 1
	integral = complex_quadrature(create_eG_integral, lower_Bound, upper_Bound, args=(H, z, w), limit = 1000)

	#int_real = integrate.quad(real_func,lower_Bound, upper_Bound, args=(H, z, w), limit=1000)[0]
	#int_imag = integrate.quad(imag_func, lower_Bound, upper_Bound, args=(H, z, w), limit=1000)[0]
	#integral = int_real + int_imag * 1j
	first_Term = 1 / (2 * numpy.pi) * integral
	second_Term = Gind(z, w)
	result = first_Term + second_Term
	return result

@jit(nopython=True)
def create_xj_vec():
	xj = numpy.zeros(pts_total)
	for i in range(pts_total):
		if i <= pts_ground - 1:
			xj[i] = i * del_width
		elif (i >= pts_ground and i <= pts_ground + pts_signal - 1):
			xj[i] = wgap + i * del_width
		else:
			xj[i] = 2 * wgap + i * del_width
	return xj

@jit(nopython=True)
def create_xi_vec():
	xi = numpy.zeros(pts_total)
	for i in range(pts_total):
		if i <= pts_ground - 1:
			xi[i] = i * del_width + distance_Antennas
		elif (i >= pts_ground and i <= pts_ground + pts_signal -1):
			xi[i] = wgap + i * del_width + distance_Antennas
		else:
			xi[i] = 2 * wgap + i * del_width + distance_Antennas
	return xi

def create_JJI_B_vecs(num_signal, num_ground, num_tot):
	B = numpy.zeros((3, num_tot), dtype = numpy.complex128)
	for i in range(num_ground):
		B[0,i] = 0
		B[0, i+num_signal+num_ground] = 0
		B[1, i] = 1
		B[1, i+num_signal+num_ground] = 0
		B[2, i] = 0
		B[2, i+num_signal+num_ground] = 1
	for i in range(num_signal):
		B[0, i+num_ground] = 1
		B[1, i+num_ground] = 0
		B[2, i+num_ground] = 0
	return B

def create_JJI_G_vecs(H, z, w, num_max):
	G = numpy.zeros((6,2 * num_max), dtype = numpy.complex128)

	for i in range(num_max):
		G[0,i] = eG(H, i * z, w)
		G[1,i] = eG(H, -i * z, w)
	
	for i in range(2 * num_max):
		G[2,i] = eG(H, i * z + wgap, w)
		G[3,i] = eG(H, -(i * z + wgap), w)
		G[4,i] = eG(H, i * z + 2 * wgap + wsignal, w)
		G[5,i] = eG(H, -(i * z + 2 * wgap + wsignal), w)
		
	return G

#def create_JJI_G_vecs(H, z, w, num_max):
#	G = numpy.zeros((6,2 * num_max), dtype = numpy.complex128)
#	args_01 = numpy.zeros((2, num_max), dtype = (numpy.complex128, 3))
#	args_2345 = numpy.zeros((4, 2 * num_max), dtype = (numpy.complex128, 3))

#	for i in range(num_max):
#		args_01[0, i] = (H, i * z, w)
#		args_01[1, i] = (H, -i * z, w)
	
#	for i in range(2 * num_max):
#		args_2345[0, i] = (H, i * z + wgap, w)
#		args_2345[1, i] = (H, -(i * z + wgap), w)
#		args_2345[2, i] = (H, i * z + 2 * wgap + wsignal, w)
#		args_2345[3, i] = (H, -(i * z + 2 * wgap + wsignal), w)
	
#	pad = numpy.zeros(num_max, dtype = numpy.complex128)
#	myPool = Pool(2)
#	G01 = numpy.zeros((2,num_max), dtype = numpy.complex128)
#	for i in range(2):
#		G01[i,:] = myPool.starmap(eG, args_01[i,:])
#		G[i,:] = numpy.append(G01[i,:],pad)
#	for i in range(4):
#		G[i+2] = myPool.starmap(eG, args_2345[i,:])
#	myPool.close
#	myPool.join
#	return G

def create_JJI_A_matrix(num_signal, num_ground, num_tot, G_vecs):
	G33_pos = G_vecs[0]     #G33 in mathcad
	G33_neg = G_vecs[1]		#G33m in mathcad
	G32_pos = G_vecs[2]		#G32 in mathcad
	G32_neg = G_vecs[3]		#G32m in mathcad
	G31_pos = G_vecs[4]		#G31 in mathcad
	G31_neg = G_vecs[5]		#G31m in mathcad

	A = numpy.zeros((num_tot, num_tot), dtype = numpy.complex128)				#changed initialising the matrix to hold 0's instead of 1's now (originally numpy.ones)
	for i in range(num_signal):
		for j in range(num_signal):
			if i - j >= 0:
				A[i+num_ground, j+num_ground] = G33_pos[i-j] * del_width
			else:
				A[i+num_ground, j+num_ground] = G33_neg[abs(i-j)] * del_width

	for i in range(num_signal):
		for j in range(num_ground):
			A[i+num_ground, j] = G32_pos[i+num_ground-1-j] * del_width
			A[i+num_ground, j+num_signal+num_ground] = G32_neg[abs(num_signal-1-i+j)] * del_width

	for i in range(num_ground):
		for j in range(num_ground):
			if i - j >= 0:
				A[i+num_ground+num_signal, j+num_ground+num_signal] = G33_pos[i-j] * del_width
				A[i, j] = G33_pos[i-j] * del_width
			else:
				A[i+num_ground+num_signal, j+num_ground+num_signal] = G33_neg[abs(i-j)] * del_width
				A[i, j] = G33_neg[abs(i-j)] * del_width

			A[i, j+num_ground+num_signal] = G31_neg[abs(num_ground-1-i+j)] * del_width
			A[i+num_ground+num_signal, j] = G31_pos[i+num_ground-1-j] * del_width

	for i in range(num_ground):
		for j in range(num_signal):
			A[i, j+num_ground] = G32_neg[abs(num_ground-1-i+j)] * del_width
			A[i+num_ground+num_signal, j+num_ground] = G32_pos[i+num_signal-1-j] * del_width

	for i in range(num_tot):
		A[i,i] = A[i,i] + var_Ys

	return A

# Current density in input antenna. Vector to store current density in individual strips of input antenna (Self consistent process, i.e. E = 1 in individual strips)
def create_JJI_ww_vecs(A, B):
	ww = numpy.zeros((3, len(A)), dtype = numpy.complex128)
	for i in range(3):
		ww[i] = numpy.linalg.solve(A, B[i])
	return ww

def create_JJI_I_matrix(ww_vecs, num_signal, num_ground, num_total):
	I = numpy.zeros((3,3), dtype = numpy.complex128)
	for i in range(3):
		I[i,0] = ww_vecs[i, num_ground : num_ground + num_signal].sum()
		I[i,1] = ww_vecs[i, 0 : num_ground].sum()
		I[i,2] = ww_vecs[i, num_ground + num_signal : num_total].sum()
	return I

def create_JJI_AL_matrix(Y11, matrix_Z):
	AL = numpy.zeros((5,5), dtype = numpy.complex128)
	AL[3,0] = -(var_R1*0 + matrix_Z[0,0] - matrix_Z[1,0])
	AL[4,0] = -(var_R1*0 + matrix_Z[0,0] - matrix_Z[2,0])
	AL[3,1] = var_R2*0 - matrix_Z[0,1] + matrix_Z[1,1]
	AL[4,1] = -(matrix_Z[0,1] - matrix_Z[2,1])
	AL[3,2] = matrix_Z[1,2] - matrix_Z[0,2]
	AL[4,2] = var_R2*0 - matrix_Z[0,2] + matrix_Z[2,2]
	AL[0,3] = -Y11/2
	AL[1,3] = Y11/2				# changed to +ve sign
	AL[0,4] = -Y11/2
	AL[2,4] = Y11/2				# changed to +ve sign

	return AL					# changed returning +ve AL (no longer return AL * (-1))


def create_JJI_bn_vec(eigVals, eigVecs):							# changed Exponents are + again. Matrix is now 5x5 added a new row and column. Second vector is simpler now (0, 0, 0, 1, 1)
	bn_first = numpy.zeros((5,5), dtype=numpy.complex128)
	bn_first[0,0] = eigVecs[0,0]+eigVecs[1,0]+eigVecs[2,0]
	bn_first[0,1] = eigVecs[0,1]+eigVecs[1,1]+eigVecs[2,1]
	bn_first[0,2] = eigVecs[0,2]+eigVecs[1,2]+eigVecs[2,2]
	bn_first[0,3] = eigVecs[0,3]+eigVecs[1,3]+eigVecs[2,3]
	bn_first[0,4] = eigVecs[0,4]+eigVecs[1,4]+eigVecs[2,4]

	bn_first[1,0] = eigVecs[3,0]
	bn_first[1,1] = eigVecs[3,1]
	bn_first[1,2] = eigVecs[3,2]
	bn_first[1,3] = eigVecs[3,3]
	bn_first[1,4] = eigVecs[3,4]

	bn_first[2,0] = eigVecs[4,0]
	bn_first[2,1] = eigVecs[4,1]
	bn_first[2,2] = eigVecs[4,2]
	bn_first[2,3] = eigVecs[4,3]
	bn_first[2,4] = eigVecs[4,4]

	bn_first[3,0] = (eigVecs[3,0]+eigVecs[1,0]*2*var_zc) * numpy.exp(eigVals[0] * length_Antenna)
	bn_first[3,1] = (eigVecs[3,1]+eigVecs[1,1]*2*var_zc) * numpy.exp(eigVals[1] * length_Antenna)
	bn_first[3,2] = (eigVecs[3,2]+eigVecs[1,2]*2*var_zc) * numpy.exp(eigVals[2] * length_Antenna)
	bn_first[3,3] = (eigVecs[3,3]+eigVecs[1,3]*2*var_zc) * numpy.exp(eigVals[3] * length_Antenna)
	bn_first[3,4] = (eigVecs[3,4]+eigVecs[1,4]*2*var_zc) * numpy.exp(eigVals[4] * length_Antenna)

	bn_first[4,0] = (eigVecs[4,0]+eigVecs[2,0]*2*var_zc) * numpy.exp(eigVals[0] * length_Antenna)
	bn_first[4,1] = (eigVecs[4,1]+eigVecs[2,1]*2*var_zc) * numpy.exp(eigVals[1] * length_Antenna)
	bn_first[4,2] = (eigVecs[4,2]+eigVecs[2,2]*2*var_zc) * numpy.exp(eigVals[2] * length_Antenna)
	bn_first[4,3] = (eigVecs[4,3]+eigVecs[2,3]*2*var_zc) * numpy.exp(eigVals[3] * length_Antenna)
	bn_first[4,4] = (eigVecs[4,4]+eigVecs[2,4]*2*var_zc) * numpy.exp(eigVals[4] * length_Antenna)

	bn_second = numpy.zeros(5, dtype = numpy.complex128)
	bn_second[0] = 0
	bn_second[1] = 0
	bn_second[2] = 0
	bn_second[3] = 2
	bn_second[4] = 2

	result = numpy.dot(numpy.linalg.inv(bn_first), bn_second)
	return result

def create_JJI_ZL_one_var(eigVals, eigVecs, bn):					# changed 4th term in numerator and denominator. Exponents also no longer are negative
	ZL_one_num = numpy.zeros(5, dtype=numpy.complex128)
	ZL_one_num[0] = eigVecs[3,0]*bn[0]*numpy.exp(length_Antenna*eigVals[0])
	ZL_one_num[1] = eigVecs[3,1]*bn[1]*numpy.exp(length_Antenna*eigVals[1])
	ZL_one_num[2] = eigVecs[3,2]*bn[2]*numpy.exp(length_Antenna*eigVals[2])
	ZL_one_num[3] = eigVecs[3,3]*bn[3]*numpy.exp(length_Antenna*eigVals[3])
	ZL_one_num[4] = eigVecs[3,4]*bn[4]*numpy.exp(length_Antenna*eigVals[4])

	ZL_one_denom = numpy.zeros(5, dtype=numpy.complex128)
	ZL_one_denom[0] = eigVecs[1,0]*bn[0]*numpy.exp(length_Antenna*eigVals[0])
	ZL_one_denom[1] = eigVecs[1,1]*bn[1]*numpy.exp(length_Antenna*eigVals[1])
	ZL_one_denom[2] = eigVecs[1,2]*bn[2]*numpy.exp(length_Antenna*eigVals[2])
	ZL_one_denom[3] = eigVecs[1,3]*bn[3]*numpy.exp(length_Antenna*eigVals[3])
	ZL_one_denom[4] = eigVecs[1,4]*bn[4]*numpy.exp(length_Antenna*eigVals[4])

	ZL_one = (ZL_one_num.sum() / ZL_one_denom.sum())
	return ZL_one

def create_JJI_ZL_two_var(eigVals, eigVecs, bn):				# changed 4th term in numerator and denominator. Exponents also no longer are negative
	ZL_two_num = numpy.zeros(5, dtype=numpy.complex128)
	ZL_two_num[0] = eigVecs[4,0]*bn[0]*numpy.exp(length_Antenna*eigVals[0])
	ZL_two_num[1] = eigVecs[4,1]*bn[1]*numpy.exp(length_Antenna*eigVals[1])
	ZL_two_num[2] = eigVecs[4,2]*bn[2]*numpy.exp(length_Antenna*eigVals[2])
	ZL_two_num[3] = eigVecs[4,3]*bn[3]*numpy.exp(length_Antenna*eigVals[3])
	ZL_two_num[4] = eigVecs[4,4]*bn[4]*numpy.exp(length_Antenna*eigVals[4])

	ZL_two_denom = numpy.zeros(5, dtype=numpy.complex128)
	ZL_two_denom[0] = eigVecs[2,0]*bn[0]*numpy.exp(length_Antenna*eigVals[0])
	ZL_two_denom[1] = eigVecs[2,1]*bn[1]*numpy.exp(length_Antenna*eigVals[1])
	ZL_two_denom[2] = eigVecs[2,2]*bn[2]*numpy.exp(length_Antenna*eigVals[2])
	ZL_two_denom[3] = eigVecs[2,3]*bn[3]*numpy.exp(length_Antenna*eigVals[3])
	ZL_two_denom[4] = eigVecs[2,4]*bn[4]*numpy.exp(length_Antenna*eigVals[4])

	ZL_two = (ZL_two_num.sum() / ZL_two_denom.sum())
	return ZL_two

def create_JJI_Ic_vec(ZL_one, ZL_two):
	Ic = numpy.zeros(3, dtype=numpy.complex128)
	Ic[0] =  2 / (2 * var_zc + ZL_one) + 2 / (2 * var_zc + ZL_two)
	Ic[1] = -2 / (2 * var_zc + ZL_one)
	Ic[2] = -2 / (2 * var_zc + ZL_two)

	return Ic

def create_JJI_b5_var(eigVals, eigVecs, bn, Ic):
	numerator = Ic[0]
	denominator = sum(eigVecs[0,0:4] * bn[0:4] * numpy.exp(length_Antenna * eigVals[0:4])) + eigVecs[0,4] * numpy.exp(length_Antenna * eigVals[4])
	#denominator = 0
	#for i in range(4):
	#	denominator += eigVecs[0,i] * bn[i] * numpy.exp(length_Antenna * eigVals[i])
	#denominator += eigVecs[0,4] * numpy.exp(length_Antenna * eigVals[4])

	return numerator / denominator

def create_JJI_Iaverage_vec(vec_Ic, eigVals, eigVecs, bn):
	Iaverage = numpy.zeros(3, dtype = numpy.complex128)
	numerator= sum(eigVecs[0,0:4]*bn[0:4] * (numpy.exp(length_Antenna * eigVals[0:4]) - 1) / (length_Antenna * eigVals[0:4]))
	numerator += eigVecs[0,4] * (numpy.exp(length_Antenna * eigVals[4]) - 1) / (length_Antenna * eigVals[4])
	denominator = sum(eigVecs[0,0:4]*bn[0:4]*numpy.exp(length_Antenna * eigVals[0:4]))
	denominator += eigVecs[0,4] * numpy.exp(length_Antenna * eigVals[4])
	#numerator = 0
	#denominator = 0
	#for i in range(4):
	#	numerator += eigVecs[0,i]*bn[i] * (numpy.exp(length_Antenna * eigVals[i]) - 1) / (length_Antenna * eigVals[i])
	#	denominator += eigVecs[0,i]*bn[i]*numpy.exp(length_Antenna * eigVals[i])
	#numerator += eigVecs[0,4] * (numpy.exp(length_Antenna * eigVals[4]) - 1) / (length_Antenna * eigVals[4])
	#denominator += eigVecs[0,4] * numpy.exp(length_Antenna * eigVals[4])
	Iaverage[0] = vec_Ic[0] * (numerator / denominator)

	numerator = sum(eigVecs[1,0:4] * bn[0:4] * (numpy.exp(length_Antenna * eigVals[0:4]) - 1) / (length_Antenna * eigVals[0:4]))
	numerator += eigVecs[1,4] * (numpy.exp(length_Antenna * eigVals[4]) - 1) / (length_Antenna * eigVals[4])
	denominator = sum(eigVecs[1,0:4]*bn[0:4] * numpy.exp(length_Antenna * eigVals[0:4]))
	denominator += eigVecs[1,4] * numpy.exp(length_Antenna * eigVals[4])
	#numerator = 0
	#denominator = 0
	#for i in range(4):
	#	numerator += eigVecs[1,i]*bn[i] * (numpy.exp(length_Antenna * eigVals[i]) - 1) / (length_Antenna * eigVals[i])
	#	denominator += eigVecs[1,i]*bn[i] * numpy.exp(length_Antenna * eigVals[i])
	#numerator += eigVecs[1,4] * (numpy.exp(length_Antenna * eigVals[4]) - 1) / (length_Antenna * eigVals[4])
	#denominator += eigVecs[1,4] * numpy.exp(length_Antenna * eigVals[4])
	Iaverage[1] = vec_Ic[1] * (numerator / denominator)

	#numerator = 0
	#denominator = 0
	#for i in range(4):
	#	numerator += eigVecs[2,i]*bn[i] * (numpy.exp(length_Antenna * eigVals[i]) - 1) / (length_Antenna * eigVals[i])
	#	denominator += eigVecs[2,i]*bn[i] * numpy.exp(length_Antenna * eigVals[i])
	numerator = sum(eigVecs[2,0:4]*bn[0:4] * (numpy.exp(length_Antenna * eigVals[0:4]) - 1) / (length_Antenna * eigVals[0:4]))
	denominator = sum(eigVecs[2,0:4]*bn[0:4] * numpy.exp(length_Antenna * eigVals[0:4]))
	numerator += eigVecs[2,4] * (numpy.exp(length_Antenna * eigVals[4]) - 1) / (length_Antenna * eigVals[4])
	denominator += eigVecs[2,4] * numpy.exp(length_Antenna * eigVals[4])
	Iaverage[2] = vec_Ic[2] * (numerator / denominator)

	return Iaverage

def create_JJI_J4average_var(eigVals, eigVecs, bn, b5):
	first_term = sum(eigVals[0:4] * eigVecs[0,0:4] * bn[0:4] * (numpy.exp(length_Antenna * eigVals[0:4]) - 1) / (length_Antenna * eigVals[0:4]))
	#for i in range(4):
	#	first_term += eigVals[i] * eigVecs[0,i] * bn[i] * (numpy.exp(length_Antenna * eigVals[i]) - 1) / (length_Antenna * eigVals[i])
	first_term += eigVals[4] * eigVecs[0,4] * (numpy.exp(length_Antenna * eigVals[4]) - 1) / (length_Antenna * eigVals[4])
	result = first_term * b5
	return result

def create_JJI_E_vec(ZZ, x, Ic):
	first_term = ZZ + x * numpy.matrix([[var_R1, 0, 0], [0, var_R2, 0], [0, 0, var_R2]])
	result = numpy.dot(first_term, Ic)
	return result

def create_JJI_B2_vec(E):
	B2 = numpy.zeros(pts_total, dtype = numpy.complex128)
	for i in range(pts_ground):
		B2[i] = E[0,1]
		B2[i+pts_signal+pts_ground] = E[0,2]
	for i in range(pts_signal):
		B2[i+pts_ground] = E[0,0]
	return B2

def create_JJI_Gout_element(i, j, H, delH, xj, xi, w):
	result = 0
	if i == 0:
		result = eG(H + delH, xi[0] - xj[j], w)
	elif j == 0:
		result = eG(H + delH, xi[i] - xj[0], w)
	elif abs((xi[i] - xj[j]) - (xi[i-1] - xj[j-1])) >= (del_width / 10):
		result = eG(H + delH, xi[i] - xj[j], w)
	return result

#def create_JJI_Gout_matrix(H, delH, xj, xi, w):
#	Gout = numpy.zeros((pts_total, pts_total), dtype = numpy.complex128)
#	xi_len = len(xi)
#	xj_len = len(xj)
#	dt = numpy.dtype([('i',numpy.int64),('j', numpy.int64),('H', numpy.complex128),('delH', numpy.complex128),('xj', numpy.complex128, xj_len), ('xi', numpy.complex128, xi_len), ('w', numpy.complex128)])
#	args = numpy.zeros((pts_total, pts_total), dtype = dt)
#	for i in range(pts_total):
#		for j in range(pts_total):
#			args[i,j] =  (i, j, H, delH, xj, xi, w)
	
#	myPool = Pool(2)
#	for i in range(pts_total):
#		Gout[:,i] = myPool.starmap(create_JJI_Gout_element, args[:,i])
#	myPool.close
#	myPool.join
#	for i in range(1, pts_total):
#		for j in range(1, pts_total):
#			if abs((xi[i] - xj[j]) - (xi[i-1] - xj[j-1])) < (del_width / 10):
#				Gout[i,j] = Gout[i-1, j-1]
#	return Gout

# The Greens function (eG) calculated at the output antenna I believe. Stored in some matrix.
def create_JJI_Gout_matrix(H, delH, xj, xi, w):											
	Gout = numpy.zeros((pts_total, pts_total), dtype = numpy.complex128)
	for i in range(pts_total):
		Gout[i,0] = eG(H + delH, xi[i] - xj[0], w)
		Gout[0,i] = eG(H + delH, xi[0] - xj[i], w)
	for i in range(1, pts_total):
		for j in range(1, pts_total):
			if abs((xi[i] - xj[j]) - (xi[i-1] - xj[j-1])) < (del_width / 10):
				Gout[i,j] = Gout[i-1, j-1]
			else:
				Gout[i,j] = eG(H + delH, xi[i] - xj[j], w)
	return Gout

# Calculate Green's function at some output distance xout for the input antenna
def create_JJI_Gout_vec(H, delH, xin, xout, w):
	Gout = numpy.zeros(len(xin), dtype = numpy.complex) 
	for i in range(len(xin)):
		Gout[i] = eG(H + delH, numpy.absolute(xout-xin[i]), w)
	return Gout

def update_JJI_B2_vec(B2, Gout, ww):
	for i in range(pts_total):
		Goutww = numpy.zeros(pts_total, dtype = numpy.complex)
		for j in range(pts_total):
			Goutww[j] = Gout[i, j] * ww[j]
		B2[i] = del_width * (numpy.sum(Goutww))
	return B2;

def create_JJI_Ix_vec(wx):
	Ix = numpy.zeros(3, dtype = numpy.complex)
	Ix[0] = del_width * (numpy.sum(wx[pts_ground:pts_ground + pts_signal - 1]))
	Ix[1] = del_width * (numpy.sum(wx[0:pts_ground - 1]))
	Ix[2] = del_width * (numpy.sum(wx[pts_ground + pts_signal : pts_total -1]))
	return Ix

def create_JJI_Diag_matrix(eigVals):
	Diag = numpy.zeros((5,5), dtype = numpy.complex128)
	for i in range(5):
		Diag[i,i] = (numpy.exp(eigVals[i] * length_Antenna) - 1) / eigVals[i]
	return Diag

def create_JJI_E2_vec(Gout, ww):
	E2 = numpy.zeros(3, dtype = numpy.complex128)
	E2[1] = 0
	E2[0] = 0
	E2[2] = 0
	for i in range(pts_ground):
		for j in range(pts_total):
			E2[1] += Gout[i,j] * ww[j]
	E2[1] = del_width * E2[1] / pts_ground

	for i in range(pts_ground, pts_ground + pts_signal):
		for j in range(pts_total):
			E2[0] += Gout[i,j] * ww[j]
	E2[0] = del_width * E2[0] / pts_signal

	for i in range(pts_ground + pts_signal, pts_total):
		for j in range(pts_total):
			E2[2] += Gout[i,j] * ww[j]
	E2[2] = del_width * E2[2] / pts_ground
	return -E2				# Changed (added minus)

# Creating Electric field distribution in output antenna. Does not average across each strip
def create_JJI_E2_vec_dist(Gout, ww):
	E2 = numpy.zeros(pts_total, dtype = numpy.complex128)
	for i in range(pts_total):
		for j in range(pts_total):
			E2[i] += Gout[i,j] * ww[j]

	return E2				# Changed (removed minus V=El)

def create_JJI_B0_vec(eigVecs, Diag, EE):
	B0 = numpy.zeros(5, dtype = numpy.complex128)
	last_vec = numpy.zeros(5, dtype = numpy.complex128)
	last_vec[3] = (EE[1] - EE[0])							# changed to +
	last_vec[4] = (EE[2] - EE[0])							# changed to +
	B0 = numpy.dot(numpy.dot(numpy.dot(eigVecs,Diag), numpy.linalg.inv(eigVecs)), last_vec)
	return B0

def create_JJI_C_matrix(eigVals, eigVecs):
	C = numpy.zeros((5,5), dtype = numpy.complex128)
	for i in range(5):
		C[0,i] = eigVecs[0,i] + eigVecs[1,i] + eigVecs[2,i]
		C[1,i] = eigVecs[3,i]
		C[2,i] = eigVecs[4,i]
		C[3,i] = (eigVecs[1,i] * 2 * var_zc + eigVecs[3,i]) * numpy.exp(eigVals[i] * length_Antenna)		
		C[4,i] = (eigVecs[2,i] * 2 * var_zc + eigVecs[4,i]) * numpy.exp(eigVals[i] * length_Antenna)		
	return C

def create_JJI_F_vec(B0):
	F = numpy.zeros(5, dtype = numpy.complex128)
	F[3] = -(B0[1] * 2 * var_zc + B0[3])			
	F[4] = -(B0[2] * 2 * var_zc + B0[4])			
	return F

def create_JJI_Iout_vec(eigVals, eigVecs, b, B0):
	Iout = numpy.zeros(3, dtype = numpy.complex128)
	for i in range(3):
		for j in range(5):
			Iout[i] += eigVecs[i,j] * b[j] * numpy.exp(eigVals[j] * length_Antenna)
		Iout[i] += B0[i]
	return Iout

def create_JJI_Vout_var(eigVals, eigVecs, b, B0):
	Vout = 0
	for i in range(5):
		Vout += eigVecs[3,i] * b[i] * numpy.exp(eigVals[i] * length_Antenna) + eigVecs[4,i] * b[i] * numpy.exp(eigVals[i] * length_Antenna)
	Vout += B0[3] + B0[4]
	Vout = Vout / 2
	return Vout

def create_JJI_Z_vec(JJI_gamma, E, Iout, Vout, ZL_one, ZL_two, j4average, w, Iaverage, Ic):
	Z = numpy.zeros(12, dtype = numpy.complex128)
	Z[0] = JJI_gamma		#S11
	Z[1] = E[0]
	Z[2] = E[1]
	Z[3] = E[2]
	Z[4] = Iout[0]
	Z[5] = Iout[1]
	Z[6] = Iout[2]
	Z[7] = Vout				#S21
	Z[8] = ZL_one
	Z[9] = ZL_two
	Z[10] = j4average / (1j * w)
	Z[11] = Iaverage[0] / Ic[0]
	return Z

def JJI(H, w, Ycss):
	var_time = time.time()
	var_delH = del_H(w)
	#print("Time: del_H(w) = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_B = create_JJI_B_vecs(pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_B_vecs = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_G = create_JJI_G_vecs(H + var_delH, del_width, w, pts_max)
	#print("Time: create_JJI_G_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_A = create_JJI_A_matrix(pts_signal, pts_ground, pts_total, vecs_JJI_G)
	#print("Time: create_JJI_A_matrix = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_ww = create_JJI_ww_vecs(matrix_JJI_A, vecs_JJI_B)
	#print("Time: create_JJI_ww_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_I = create_JJI_I_matrix(vecs_JJI_ww, pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_I_matrix = ", time.time() - var_time)

	matrix_JJI_ZZ = numpy.linalg.inv(numpy.transpose(matrix_JJI_I)) / del_width   #Changed (removed minus since electronics engineering uses this convention. V=El) 
	var_Y11 = Ycss / 2 * w
	#var_time = time.time()
	matrix_JJI_AL = create_JJI_AL_matrix(var_Y11, matrix_JJI_ZZ)
	#print("Time: create_JJI_AL_matrix = ", time.time() - var_time)

	vec_eigAL = numpy.linalg.eig(matrix_JJI_AL)[0]
	vec_eigVecAL = numpy.linalg.eig(matrix_JJI_AL)[1]

	vec_JJI_bn = create_JJI_bn_vec(vec_eigAL, vec_eigVecAL)
	var_JJI_ZL_one = create_JJI_ZL_one_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_ZL_two = create_JJI_ZL_two_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_ZL = var_JJI_ZL_one * var_JJI_ZL_two / (var_JJI_ZL_one + var_JJI_ZL_two)
	var_JJI_gamma = (var_JJI_ZL - var_zc) / (var_JJI_ZL + var_zc)						# S11
	vec_JJI_Ic = create_JJI_Ic_vec(var_JJI_ZL_one, var_JJI_ZL_two)
	var_JJI_b5 = create_JJI_b5_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn, vec_JJI_Ic)
	vec_JJI_Iaverage = create_JJI_Iaverage_vec(vec_JJI_Ic, vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_J4average = create_JJI_J4average_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn, var_JJI_b5)
	vec_JJI_E = create_JJI_E_vec(matrix_JJI_ZZ, 0, vec_JJI_Ic)
	vec_JJI_B2 = create_JJI_B2_vec(vec_JJI_E)
	vec_JJI_ww2 = numpy.linalg.solve(matrix_JJI_A, vec_JJI_B2)
	vec_JJI_xi = create_xi_vec()
	vec_JJI_xj = create_xj_vec()
	matrix_JJI_Gout = create_JJI_Gout_matrix(H, var_delH, vec_JJI_xj, vec_JJI_xi, w)
	vec_JJI_B2 = update_JJI_B2_vec(vec_JJI_B2, matrix_JJI_Gout, vec_JJI_ww2)
	vec_JJI_wx = numpy.linalg.solve(matrix_JJI_A, vec_JJI_B2)
	vec_JJI_Ix = create_JJI_Ix_vec(vec_JJI_wx)

	vec_JJI_EE = numpy.dot(matrix_JJI_ZZ, vec_JJI_Ix)						# Electric field in output antenna, indices correspond to signal, and 2 ground lines of antenna
	matrix_JJI_Diag = create_JJI_Diag_matrix(vec_eigAL)
	vec_JJI_B0 = create_JJI_B0_vec(vec_eigVecAL, matrix_JJI_Diag, vec_JJI_EE)
	matrix_JJI_C = create_JJI_C_matrix(vec_eigAL, vec_eigVecAL)
	vec_JJI_F = create_JJI_F_vec(vec_JJI_B0)
	var_JJI_b = numpy.dot(numpy.linalg.inv(matrix_JJI_C), vec_JJI_F)
	vec_JJI_Iout = create_JJI_Iout_vec(vec_eigAL, vec_eigVecAL, var_JJI_b, vec_JJI_B0)
	var_JJI_Vout = create_JJI_Vout_var(vec_eigAL, vec_eigVecAL, var_JJI_b, vec_JJI_B0)
	vec_JJI_Z = create_JJI_Z_vec(var_JJI_gamma, vec_JJI_EE, vec_JJI_Iout, var_JJI_Vout, var_JJI_ZL_one, var_JJI_ZL_two, var_JJI_J4average, w, vec_JJI_Iaverage, vec_JJI_Ic)
	
	#print(numpy.shape(vec_JJI_ww2), pts_total)
	print("Frequency: ", numpy.real(w/(2*numpy.pi)))
	print("Time Frequency: ", time.time() - var_time)
	#print("vec_JJI_Z: ")
	#print(vec_JJI_Z)
	#print(vec_JJI_ww2)
	final_result = numpy.concatenate((vec_JJI_Z, vec_JJI_ww2, matrix_JJI_ZZ.flatten()))    #returnin Z vector, complex current density, and impedance ZZ matrix
	#print(numpy.shape(final_result))
		
	return final_result     #changed returning both the Z vector as well as the complex current density WW2, and impedance ZZ matrix


# This function is used only in testing when wanting to obtain current density in single strips when E = 1 in individual strips
def JJI_ww_single(H, w, Ycss):
	var_time = time.time()
	var_delH = del_H(w)
	#print("Time: del_H(w) = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_B = create_JJI_B_vecs(pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_B_vecs = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_G = create_JJI_G_vecs(H + var_delH, del_width, w, pts_max)
	#print("Time: create_JJI_G_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_A = create_JJI_A_matrix(pts_signal, pts_ground, pts_total, vecs_JJI_G)
	#print("Time: create_JJI_A_matrix = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_ww = create_JJI_ww_vecs(matrix_JJI_A, vecs_JJI_B)
	#print("Time: create_JJI_ww_vecs = ", time.time() - var_time)

	return vecs_JJI_ww     #this is the extra JJI that returns the current density in a single strip earlier in code


#test function
def JJI_E2_distribution_outantenna(H, w, Ycss):
	var_time = time.time()
	var_delH = del_H(w)
	#print("Time: del_H(w) = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_B = create_JJI_B_vecs(pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_B_vecs = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_G = create_JJI_G_vecs(H + var_delH, del_width, w, pts_max)
	#print("Time: create_JJI_G_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_A = create_JJI_A_matrix(pts_signal, pts_ground, pts_total, vecs_JJI_G)
	#print("Time: create_JJI_A_matrix = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_ww = create_JJI_ww_vecs(matrix_JJI_A, vecs_JJI_B)
	#print("Time: create_JJI_ww_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_I = create_JJI_I_matrix(vecs_JJI_ww, pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_I_matrix = ", time.time() - var_time)

	matrix_JJI_ZZ = numpy.linalg.inv(numpy.transpose(matrix_JJI_I)) / del_width   #Changed (added minus)
	var_Y11 = Ycss * w
	#var_time = time.time()
	matrix_JJI_AL = create_JJI_AL_matrix(var_Y11, matrix_JJI_ZZ)
	#print("Time: create_JJI_AL_matrix = ", time.time() - var_time)

	vec_eigAL = numpy.linalg.eig(matrix_JJI_AL)[0]
	vec_eigVecAL = numpy.linalg.eig(matrix_JJI_AL)[1]

	vec_JJI_bn = create_JJI_bn_vec(vec_eigAL, vec_eigVecAL)
	var_JJI_ZL_one = create_JJI_ZL_one_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_ZL_two = create_JJI_ZL_two_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_ZL = var_JJI_ZL_one * var_JJI_ZL_two / (var_JJI_ZL_one + var_JJI_ZL_two)
	var_JJI_gamma = (var_JJI_ZL - var_zc) / (var_JJI_ZL + var_zc)						# S11
	vec_JJI_Ic = create_JJI_Ic_vec(var_JJI_ZL_one, var_JJI_ZL_two)
	var_JJI_b5 = create_JJI_b5_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn, vec_JJI_Ic)
	vec_JJI_Iaverage = create_JJI_Iaverage_vec(vec_JJI_Ic, vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_J4average = create_JJI_J4average_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn, var_JJI_b5)
	vec_JJI_E = create_JJI_E_vec(matrix_JJI_ZZ, 0, vec_JJI_Ic)
	vec_JJI_B2 = create_JJI_B2_vec(vec_JJI_E)
	vec_JJI_ww2 = numpy.linalg.solve(matrix_JJI_A, vec_JJI_B2)
	vec_JJI_xi = create_xi_vec()
	vec_JJI_xj = create_xj_vec()
	matrix_JJI_Gout = create_JJI_Gout_matrix(H, var_delH, vec_JJI_xj, vec_JJI_xi, w)

	vec_JJI_E2 = create_JJI_E2_vec_dist(matrix_JJI_Gout, vec_JJI_ww2)						# Electric field in output antenna, indices correspond to signal, and 2 ground lines of antenna
	
	return vec_JJI_E2     #changed returning both the Z vector as well as the complex current density WW2

def JJI_Edist_custom(H, w, Ycss, xout):
	print("Enter: JJI_Edist_custom")
	var_time = time.time()
	var_delH = del_H(w)
	#print("Time: del_H(w) = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_B = create_JJI_B_vecs(pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_B_vecs = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_G = create_JJI_G_vecs(H + var_delH, del_width, w, pts_max)
	#print("Time: create_JJI_G_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_A = create_JJI_A_matrix(pts_signal, pts_ground, pts_total, vecs_JJI_G)
	#print("Time: create_JJI_A_matrix = ", time.time() - var_time)

	#var_time = time.time()
	vecs_JJI_ww = create_JJI_ww_vecs(matrix_JJI_A, vecs_JJI_B)
	#print("Time: create_JJI_ww_vecs = ", time.time() - var_time)

	#var_time = time.time()
	matrix_JJI_I = create_JJI_I_matrix(vecs_JJI_ww, pts_signal, pts_ground, pts_total)
	#print("Time: create_JJI_I_matrix = ", time.time() - var_time)

	matrix_JJI_ZZ = numpy.linalg.inv(numpy.transpose(matrix_JJI_I)) / del_width   #Changed removed minus
	var_Y11 = Ycss * w
	#var_time = time.time()
	matrix_JJI_AL = create_JJI_AL_matrix(var_Y11, matrix_JJI_ZZ)
	#print("Time: create_JJI_AL_matrix = ", time.time() - var_time)

	vec_eigAL = numpy.linalg.eig(matrix_JJI_AL)[0]
	vec_eigVecAL = numpy.linalg.eig(matrix_JJI_AL)[1]

	vec_JJI_bn = create_JJI_bn_vec(vec_eigAL, vec_eigVecAL)
	var_JJI_ZL_one = create_JJI_ZL_one_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_ZL_two = create_JJI_ZL_two_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_ZL = var_JJI_ZL_one * var_JJI_ZL_two / (var_JJI_ZL_one + var_JJI_ZL_two)
	var_JJI_gamma = (var_JJI_ZL - var_zc) / (var_JJI_ZL + var_zc)						# S11
	vec_JJI_Ic = create_JJI_Ic_vec(var_JJI_ZL_one, var_JJI_ZL_two)
	var_JJI_b5 = create_JJI_b5_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn, vec_JJI_Ic)
	vec_JJI_Iaverage = create_JJI_Iaverage_vec(vec_JJI_Ic, vec_eigAL, vec_eigVecAL, vec_JJI_bn)
	var_JJI_J4average = create_JJI_J4average_var(vec_eigAL, vec_eigVecAL, vec_JJI_bn, var_JJI_b5)
	vec_JJI_E = create_JJI_E_vec(matrix_JJI_ZZ, 0, vec_JJI_Ic)
	vec_JJI_B2 = create_JJI_B2_vec(vec_JJI_E)
	vec_JJI_ww2 = numpy.linalg.solve(matrix_JJI_A, vec_JJI_B2)
	vec_JJI_xj = create_xj_vec()

	print("Starting custom E field calculations")
	var_time = time.time()
	green_dist = numpy.zeros(len(xout), dtype = numpy.complex)
	for i in range(len(xout)):
		print("xval: ", xout[i])
		t1 = time.time()
		green_i = create_JJI_Gout_vec(H, var_delH, vec_JJI_xj, xout[i], w)
		green_dist[i] = -1*numpy.sum(numpy.multiply(green_i, vec_JJI_ww2))
		print("t: ", time.time() - t1)
	print("Time for all distances: ", time.time()-var_time)
		
	print("Exiting: JJI_Edist_custom")
	return green_dist     #changed returning both the Z vector as well as the complex current density WW2

def create_main_freq_vec(freq_lower, freq_step, plot_pts_num):
	result = numpy.zeros(plot_pts_num, numpy.float64)
	for i in range(plot_pts_num):
		result[i] = freq_lower + i * freq_step
	return result

#def create_plots(result_matrix, frequencies):
#	vec_s21 = numpy.zeros(len(frequencies), dtype = numpy.complex128)
#	for i in range(len(frequencies)):
#		vec_s21[i] = result_matrix[i][7]
#	plt.plot(frequencies / (2 * numpy.pi), numpy.abs(vec_s21), 'ro')
#	plt.show()
#	return 0

#updated savefile saving all parameters		
def create_savefile(filename, result_matrix, frequencies):
	matrix = numpy.zeros((len(frequencies), 50), dtype = numpy.float64)								#changed increased to 50 to store unwrapped phase as well
	current_densities = numpy.zeros((len(frequencies), 4*pts_total + 1), dtype = numpy.float64)
	impedance = numpy.zeros((len(frequencies), 9*4 + 1), dtype=numpy.float64)
	for i in range(len(frequencies)):
		#This was how we had it originally
		matrix[i,0] = frequencies[i] / (2 * numpy.pi)
		current_densities[i,0] = frequencies[i] / (2*numpy.pi)

		impedance[i,0] = frequencies[i] / (2*numpy.pi)
		#matrix[i,1] = numpy.real(result_matrix[i][7])
		#matrix[i,2] = numpy.imag(result_matrix[i][7])
		#matrix[i,3] = numpy.real(result_matrix[i][0])
		#matrix[i,4] = numpy.imag(result_matrix[i][0])
		#matrix[i,5] = numpy.absolute(result_matrix[i][7])
		#matrix[i,6] = numpy.absolute(result_matrix[i][0])
		#matrix[i,7] = numpy.angle(result_matrix[i][7])
		#matrix[i,8] = numpy.angle(result_matrix[i][0])

		#We will now save all of the results not just the S parameter
		for j in range(12):
			matrix[i,4*j+1] = numpy.real(result_matrix[i][j])
			matrix[i,4*j+2] = numpy.imag(result_matrix[i][j])
			matrix[i,4*j+3] = numpy.absolute(result_matrix[i][j])
			matrix[i,4*j+4] = numpy.angle(result_matrix[i][j])


		for j in range(pts_total):   #changed
			current_densities[i,j+1] = numpy.real(result_matrix[i][j+12])
			current_densities[i,j+pts_total+1] = numpy.imag(result_matrix[i][j+12])
			current_densities[i,j+2*pts_total+1] = numpy.absolute(result_matrix[i][j+12])
			current_densities[i,j+3*pts_total+1] = numpy.angle(result_matrix[i][j+12])

		for j in range(9):
			impedance[i, 4*j+1] = numpy.real(result_matrix[i][j+12+pts_total])
			impedance[i, 4*j+2] = numpy.imag(result_matrix[i][j+12+pts_total])
			impedance[i, 4*j+3] = numpy.absolute(result_matrix[i][j+12+pts_total])
			impedance[i, 4*j+4] = numpy.angle(result_matrix[i][j+12+pts_total])
		

	matrix[:,49] = numpy.unwrap(matrix[:,32])					#changed adding unwrapped phase to output data
	
	head = "Frequency"
	for i in range(3):
		for j in range(3):
			head += ", Real(Z"+str(i+1)+str(j+1)+"), Imag(Z"+str(i+1)+str(j+1)+"), Amp(Z"+str(i+1)+str(j+1)+"), Phase(Z"+str(i+1)+str(j+1)+")"  
	
	numpy.savetxt(filename + ".csv", matrix, header = "Frequency, Real(S11), Imag(S11), Amp(S11), Phase(S11), Real(E0), Imag(E0), Amp(E0), Phase(E0), Real(E1), Imag(E1), Amp(E1), Phase(E1), Real(E2), Imag(E2), Amp(E2), Phase(E2), Real(I0), Imag(I0), Amp(I0), Phase(I0), Real(I1), Imag(I1), Amp(I1), Phase(I1), Real(I2), Imag(I2), Amp(I2), Phase(I2), Real(S21), Imag(S21), Amp(S21), Phase(S21), Real(ZL1), Imag(ZL1), Amp(ZL1), Phase(ZL1), Real(ZL2), Imag(ZL2), Amp(ZL2), Phase(ZL2), Real(J4avg), Imag(J4avg), Amp(J4avg), Phase(J4avg), Real(Icavg), Imag(Icavg), Amp(Icavg), Phase(Icavg), Unwrapped Phase(S21) ", delimiter=',')	#changed added Unwrapped Phase(S21) to header
	numpy.savetxt(filename + "_vars.txt", create_global_vars_matrix(), fmt = "%s")
	numpy.savetxt(filename + "_current.csv", current_densities, header = "Frequency, Real(ww2) all, Imag(ww2) all, Amp(ww2) all, Phase(ww2) all", delimiter= ',')    #changed 
	numpy.savetxt(filename + "_impedance.csv", impedance, header= head, delimiter=',')  #changed
	return 0

def simulate(current_pool, Ycss, start_freq, end_freq, points, filename):
	vec_frequencies = create_main_freq_vec(start_freq, freq_step, points)
	print("Frequencies: ", vec_frequencies/(2*numpy.pi))
	result_matrix = numpy.zeros((len_JJI_Z_vec, points), dtype = numpy.complex128)
	
	vec_arguments = numpy.zeros(points, dtype = (numpy.complex128, 3))

	for i in range(points):
		vec_arguments[i] = (Hi, vec_frequencies[i], Ycss)
	result_matrix = current_pool.starmap(JJI, vec_arguments)      #This seems to transpose result_matrix from (x,y) to (y,x)
	#print("Result: ")
	#print(result_matrix)
	#print("Result[0][1]: ", result_matrix[0][1])
	#print("Result[0]: ", result_matrix[0])
	#print("Result[:][0]: ", result_matrix[:][0])
	#create_plots(result_matrix, vec_frequencies)
	create_savefile(filename, result_matrix, vec_frequencies)
	return 0

def recompile_funcs():
	ww.recompile()
	Q2.recompile()
	Q4.recompile()
	QQ.recompile()
	S0h.recompile()
	SLj.recompile()
	SLh.recompile()
	Cmy.recompile()
	Chx.recompile()
	#Chy.recompile()         #isn't currently used therefore no need to recompile
	create_b_var.recompile()
	create_c_var.recompile()
	create_d_var.recompile()
	create_Cmy_vec.recompile()
	create_Chx_vec.recompile()
	create_A_matrix.recompile()
	create_B_matrix.recompile()
	create_M_vec.recompile()
	create_hxk_var.recompile()
	create_hyl_var.recompile()
	MM.recompile()
	create_Gind_integral.recompile()
	create_eG_integral.recompile()
	create_xj_vec.recompile()
	create_xi_vec.recompile()
	return 0

def sim_varying_args(dir, Ycss, lowB, upB, pts, arg, **kwargs):
	arg_val = lowB
	step = 0
	if (pts > 1):
		step = (upB - lowB)/(pts-1)
	total_time = 0
	for i in range(pts):
		var_time = time.time()
		append_id = ""
		if ('id' in kwargs):					#changed to store id at end of filename
			append_id = "_" + kwargs.get("id")
		name = os.path.join(dir, str(date.today()) + "_" + arg + "_" + str(arg_val) + append_id)
		#arg_dict = (arg, arg_val)
		#update_dict_vars(ind_Variables, arg_dict)
		arg_dict = {arg:arg_val}	#changed
		arg_dict.update(kwargs)    #changed
		update_dict_vars(ind_Variables, arg_dict)   #changed
		recompile_funcs()
		myPool = Pool(4, initializer=update_dict_vars, initargs = (ind_Variables, arg_dict))
		simulate(myPool, Ycss, freq_lower, freq_upper, plot_pts_num, name)
		myPool.close()
		myPool.join()
		arg_val += step
		time_simulation = time.time() - var_time
		total_time += time_simulation
		print("Time Simulation: ", time_simulation)
	return total_time / (pts)

def mm_plot(xvals, frequency,**kwargs):
	yvals = numpy.zeros(len(xvals), dtype=numpy.float64)
	update_dict_vars(ind_Variables, kwargs)
	for i in range(len(xvals)):
		yvals[i] = math.log(abs(MM(xvals[i],gamma*(Hi + del_H(frequency)*0), frequency)))    #changed reduced the losses
	return yvals

def mm_plot_frequency(xfreqs,k,**kwargs):
	yvals = numpy.zeros(len(xfreqs), dtype=numpy.float64)
	update_dict_vars(ind_Variables, kwargs)
	for i in range(len(xfreqs)):
		yvals[i] = (abs(MM(k,gamma*(Hi+del_H(xfreqs[i])),xfreqs[i])))
	return yvals

def eG_plot(H, xvals, frequency,**kwargs):
	yvals = numpy.zeros(len(xvals), dtype=numpy.complex)
	update_dict_vars(ind_Variables, kwargs)
	for i in range(len(xvals)):
		yvals[i] = eG(H, xvals[i], frequency)
	return yvals

def curr_single_strip(H, w, Ycss, strip):
	curr = JJI_ww_single(H, w, Ycss)[strip]
	print(numpy.shape(curr))
	length = len(curr)
	x_dist = numpy.zeros(length, dtype = numpy.float)

	for i in range(length):
		x_dist[i] = i * del_width

	out_matrix = x_dist
	out_matrix = numpy.column_stack((out_matrix, numpy.real(curr)))
	out_matrix = numpy.column_stack((out_matrix, numpy.imag(curr)))
	out_matrix = numpy.column_stack((out_matrix, numpy.absolute(curr)))
	out_matrix = numpy.column_stack((out_matrix, numpy.angle(curr)))

	directory = 'H:/My Documents/Physics/PhD Work/Simulation Code/Python Simulation Results/Output Tests'
	file = 'Jcurr_self_' + str(strip) + '_' + str(w/(2*numpy.pi))
	file_tosave = os.path.join(directory, file)

	head = "x, Re(J), Im(J), |J|, arg(J)"
	numpy.savetxt(file_tosave + ".csv", out_matrix, delimiter = ',', header = head)
	
	return 0

def Efield_dist(H, w, Ycss):
	efield = JJI_E2_distribution_outantenna(H, w, Ycss)
	x_dist = numpy.zeros(len(efield), dtype = numpy.float)

	for i in range(pts_total):
		x_dist[i] = i * del_width

	out_matrix = x_dist
	out_matrix = numpy.column_stack((out_matrix, numpy.real(efield)))
	out_matrix = numpy.column_stack((out_matrix, numpy.imag(efield)))
	out_matrix = numpy.column_stack((out_matrix, numpy.absolute(efield)))
	out_matrix = numpy.column_stack((out_matrix, numpy.angle(efield)))

	directory = 'H:/My Documents/Physics/PhD Work/Simulation Code/Python Simulation Results/Output Tests'
	file = 'Efield_outantenna_onband' + str(w/(2*numpy.pi))
	file_tosave = os.path.join(directory, file)

	head = "x, Re(E), Im(E), |E|, arg(E)"
	numpy.savetxt(file_tosave + ".csv", out_matrix, delimiter = ',', header = head)
	
	return 0

def Efield_dist_custom(H, w, Ycss, xout):
	print("Enter: Efield_dist_custom")
	efield = JJI_Edist_custom(H, w, Ycss, xout)
	
	out_matrix = xout
	out_matrix = numpy.column_stack((out_matrix, numpy.real(efield)))
	out_matrix = numpy.column_stack((out_matrix, numpy.imag(efield)))
	out_matrix = numpy.column_stack((out_matrix, numpy.absolute(efield)))
	out_matrix = numpy.column_stack((out_matrix, numpy.unwrap(numpy.angle(efield))))

	directory = 'H:/My Documents/Physics/PhD Work/Simulation Code/Python Simulation Results/20210623_Conductivity/ElectricField'
	file = 'Efield_x-distance_0.5umstep_' + str(w/(2*numpy.pi))
	file_tosave = os.path.join(directory, file)

	head = "x, Re(E), Im(E), |E|, arg(E)"
	numpy.savetxt(file_tosave + ".csv", out_matrix, delimiter = ',', header = head)
	
	print("Exiting: Efield_dist_custom")
	return 0

def sim_varying_args(dir, Ycss, lowB, upB, pts, arg, **kwargs):
	arg_val = lowB
	step = 0
	if (pts > 1):
		step = (upB - lowB)/(pts-1)
	total_time = 0
	for i in range(pts):
		var_time = time.time()
		append_id = ""
		if('id' in kwargs):
			append_id = "_" + kwargs.get("id")
		name = os.path.join(dir, str(date.today()) + "_" + arg + "_" + str(arg_val) + append_id)
		# arg_dict = (arg, arg_val)
		arg_dict = {arg:arg_val}
		arg_dict.update(kwargs)
		update_dict_vars(ind_Variables, arg_dict)
		recompile_funcs()
		myPool = Pool(4, initializer=update_dict_vars, initargs = (ind_Variables, arg_dict))
		simulate(myPool, Ycss, freq_lower, freq_upper, plot_pts_num, name)
		myPool.close()
		myPool.join()
		arg_val += step
		time_simulation = time.time() - var_time
		total_time += time_simulation
		print("Time Simulation: ", time.time() - var_time)
	return 0

def main():
	directory = "C:\source\SimulationResults\SupplementaryMaterials"
	print("Start: antennaCalcs()")
	var_time = time.time()
	var_Ycss = antennaCalcs()
	print(var_Ycss)
	average_simulation_time = sim_varying_args(directory, var_Ycss, 3*10**-3, 3*10**-3, 1, 'surface_Ks1', id="corrected_+k_Ks2=0")
	total_time = time.time() - var_time
	numpy.savetxt(os.path.join(directory, "times.csv"), numpy.array(((average_simulation_time, total_time),)), delimiter = ",", header = "Average Simulation Time, Total Time")

	
	
	

	#t = time.time()
	#print("Start: ", t)
	#freq = 17.886*10**9 * 2*numpy.pi
	#####print(Gind(0.1*10**-6, freq))
	#####print(MM(1,1,1))
	#####print(MM(3.8*10**6,gamma*(appliedH+del_H(centralFreq)),centralFreq))
	#xvals = numpy.arange(0.1*10**6,10*10**6,1*10**4,dtype=numpy.float)
	#xdist = numpy.arange(0.1*10**-6, 5*10**-6, 0.1*10**-6, dtype=numpy.float)
	####xfreqs=2*numpy.pi*numpy.arange(4*10**9,18*10**9,0.1*10**8,dtype=numpy.float)
	###xfreqsplot = xfreqs / (2*numpy.pi)
	#y1 = mm_plot(xvals, freq)
	####print(appliedH, Hi, omegaU, omegaH, Hubx, satMs, pinning_d1y, pinning_d2y)
	#y2 = mm_plot(xvals,freq, appliedH=-appliedH)
	###print(appliedH, Hi, omegaU, omegaH, Hubx, satMs, pinning_d1y, pinning_d2y)
	###y3 = mm_plot(xvals, freq, appliedH = -500*79.57747, surface_Ks1 = 2.25*10**-3)
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	###y4 = mm_plot(xvals, freq, appliedH = 500*79.57747, surface_Ks2 = 2.25*10**-3)
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	###y5 = mm_plot(xvals, freq, appliedH = -500*79.57747, surface_Ks2 = 2.25*10**-3)
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	###y1freqs = mm_plot_frequency(xfreqs,7.8*10**6, appliedH = 500*79.57747 + del_H(freq))
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	###y2freqs = mm_plot_frequency(xfreqs,3.6*10**6, appliedH = 230*79.57747, surface_Ks1=2.25*10**-3)
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	###y3freqs = mm_plot_frequency(xfreqs,3.6*10**6, appliedH = -230*79.57747, surface_Ks1=2.25*10**-3)
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	###y4freqs = mm_plot_frequency(xfreqs,3.6*10**6, surface_Ks1=0*2*10**-3, surface_Ks2=0*0.25*10**-3)
	###print(appliedH, satMs, pinning_d1y, pinning_d2y)
	#plt.plot(xvals,y1,'b-')
	#plt.plot(xvals,y2,'r-')
	####plt.plot(xvals,y3,'g-')
	####plt.plot(xvals,y4,'y-')
	####plt.plot(xvals,y5,'c-')
	####plt.plot(xfreqs,y1freqs,xfreqs,y2freqs,xfreqs,y3freqs)
	####plt.plot(xfreqsplot, y1freqs, 'b-')
	####plt.plot(xfreqsplot, y2freqs, 'r-')
	####plt.plot(xfreqsplot, y3freqs, 'g-')
	####plt.plot(xfreqsplot, y4freqs, 'y-')
	#plt.show()

	#directory = 'H:/My Documents/Physics/PhD Work/Simulation Code/Python Simulation Results/20210623_Conductivity/ElectricField'
	#file = 'Greensfunction_'+str(freq / (2*numpy.pi*10**9))+'GHz'
	#file_tosave = os.path.join(directory, file)

	#xout = xvals
	#yout = y1
	#outmatrix = numpy.column_stack((xout, y1))

	#def egVec(z, H, w):
	#	return eG(H, z, w)

	#veceG = numpy.vectorize(egVec)


	#xout = xdist
	#yout = veceG(xdist, Hi + del_H(freq), freq)
	#outmatrix = numpy.column_stack((xout, numpy.absolute(yout), numpy.unwrap(numpy.angle(yout))))

	


	####y = eG_plot(Hi+del_H(freq), xdist, freq)
	####print("Gind: ", Gind(1*10**-6,freq))
	####plt.plot(xdist, numpy.absolute(y), 'b-')
	#plt.show()


	#numpy.savetxt(file_tosave + ".csv",outmatrix, delimiter=',', header = "k, MM")
	#numpy.savetxt(file_tosave + "_vars.txt", create_global_vars_matrix(), fmt = "%s")
	##print(eG(Hi+del_H(freq), 1*10**-5, freq))
	#print("Finish: ", time.time()-t)

	####curr_single_strip(Hi, freq, var_Ycss, 0)
	#xvals = numpy.zeros(50, dtype=numpy.float)
	###xoutantenna = create_xi_vec()
	#for i in range(len(xvals)):
	#	xvals[i] = i*0.5*10**-6 + 2*10**-6
	##print("xvals: ", xvals)
	#Efield_dist_custom(Hi, freq, var_Ycss, xvals)
	##Efield_dist(Hi, freq, var_Ycss)

	return 0

if __name__ == '__main__':
	#cProfile.run('main()', 'profile_stats')
	#p = pstats.Stats('profile_stats')
	#p.strip_dirs().sort_stats('file').print_stats()
	#p.sort_stats('time').print_stats()
	main()
