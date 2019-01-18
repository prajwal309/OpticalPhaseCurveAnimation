#Author:Prajwal Niraula
#Institute: EAPS, MIT

import numpy as np
import mayavi.mlab as mlab
import  moviepy.editor as mpy
import batman
#Create the Surface of the Sphere


#Create a spherical object
RStar = 1.0         #Radius of the star
RPlanet = 0.2       #Radius of the planet
u1 = 0.5            #limb darkening parameter
a_R = 5.0           #Semi Scaled angle
TDur = 7.5          #Duration
FPS = 30            #Frames per Second
MassRatio = 0.05    #Mass ratio of planet and star

#Generate the light curve

#Generate Transit parameters
params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 1.                       #orbital period
params.rp = 0.003                     #planet radius (in units of stellar radii)
params.a = 5.0                        #semi-major axis (in units of stellar radii)
params.inc = 87.                      #orbital inclination (in degrees)
params.ecc = 0.                       #eccentricity
params.w = 90.                        #longitude of periastron (in degrees)
params.limb_dark = "linear"           #limb darkening model
params.u = [u1]                       #limb darkening coefficients [u1]

#Generating the transits
phase = np.linspace(0.90, 2.10, 5000)  #times at which to calculate light curve
NormFactor = 1.0/(max(phase)-min(phase)) #Normalization factor

TransitLC = batman.TransitModel(params, phase, exp_time=0.020)    #initializes model


#Generating the phase curve signal
Ref= 2.0+2.0*np.cos(2.0*np.pi*phase-np.pi)    #Reflective and thermal component
Ell= 1.*np.cos(4.0*np.pi*phase+np.pi)       #Ellipsoidal component
Dop = 0.7*np.sin(2.0*np.pi*phase)          #Doppler component


#Generating the occultation
params.fp = 4.0
params.t_secondary = 0.5
OccultationLC = batman.TransitModel(params, phase, transittype="secondary", exp_time=0.02)

LC = (TransitLC.light_curve(params)-1.0)*1.e6 + Ref + Ell + Dop + (OccultationLC.light_curve(params)-50.0)

#Phi goes from 0 to pi, Theta goes from 0 to 2 pi
Resolution = 100
Phi, Theta = np.meshgrid(np.linspace(0,np.pi,Resolution),np.linspace(0,2.0*np.pi,Resolution))


XPlanet = RPlanet*np.sin(Phi)*np.cos(Theta)
YPlanet = RPlanet*np.sin(Phi)*np.sin(Theta)
ZPlanet = RPlanet*np.cos(Phi)

RDot = 0.2
XDot = RDot*np.sin(Phi)*np.cos(Theta)
YDot = RDot*np.sin(Phi)*np.sin(Theta)
ZDot = RDot*np.cos(Phi)


#Generate the light curve using batman

def make_frame(t):
    PhaseShift = 0.90*np.pi
    Angle = t/TDur*2.0*np.pi + PhaseShift
    XLoc = a_R*np.sin(Angle)
    ZLoc = a_R*np.cos(Angle)

    SFact = 4.0
    XStretchFactor = (SFact+np.abs(np.cos(PhaseShift)))/SFact
    ZStretchFactor = (SFact+np.abs(np.sin(PhaseShift)))/SFact
    XStar = XStretchFactor*RStar*np.sin(Phi)*np.cos(Theta)
    YStar = RStar*np.sin(Phi)*np.sin(Theta)
    ZStar = ZStretchFactor*RStar*np.cos(Phi)

    #Location of the star
    XStar_Loc = MassRatio*a_R*XLoc
    ZStar_Loc = MassRatio*a_R*ZLoc


    #See which angle faces the star vs which is away from it
    PlanetSurface = a_R + np.sqrt((XPlanet-XLoc)**2.0+ YPlanet**2.0+ (ZPlanet-ZLoc)**2.0)

    #Construct Stellar Surface
    StellarSurface = ((PlanetSurface-np.mean(PlanetSurface))**2.0)**0.998

    #Plot the star and the planet
    mlab.clf()

    #Offset for Y
    FigureYOffSet = 8.5

    #Find the corresponding point in the LC
    PhaseCalc = (Angle - PhaseShift)/(2.0*np.pi)
    Shift = -0.050                                               #Shift at the beginning

    #Showing the mesh of the star
    if np.abs(PhaseCalc+Shift-0.25)<0.25:
        mlab.mesh(XStar+XStar_Loc , YStar+FigureYOffSet, ZStar+ZStar_Loc, scalars=StellarSurface, colormap='Blues')
    else:
        mlab.mesh(XStar+XStar_Loc , YStar+FigureYOffSet, ZStar+ZStar_Loc, scalars=StellarSurface, colormap='Reds')

    #showing the mesh of the planet
    mlab.mesh(XPlanet-XLoc, YPlanet+FigureYOffSet, ZPlanet-ZLoc,scalars=PlanetSurface,colormap='BuGn')

    OffsetX = 0.25
    OffsetY = -5.0

    def ProcessData(XArray, YArray):
        return XArray - np.mean(XArray) - OffsetX, 0.25*(YArray - np.mean(YArray)) - OffsetY

    #Coordinates to plot LC
    PhaseX = 2.0*phase*a_R
    PhaseX, PhaseY = ProcessData(PhaseX, LC)
    _,  NewRef = ProcessData(PhaseX, Ref)
    _,  NewEll = ProcessData(PhaseX, Ell)
    _,  NewDop = ProcessData(PhaseX, Dop)

    mlab.plot3d(PhaseX, PhaseY, np.zeros(len(phase)),   np.zeros(len(phase)), tube_radius=0.03, colormap='Blues')
    mlab.plot3d(PhaseX, NewRef+0.1, np.zeros(len(phase)),   np.zeros(len(phase)), tube_radius=0.025, colormap='ocean')
    mlab.plot3d(PhaseX, NewEll-0.1, np.zeros(len(phase)),   np.ones(len(phase)), tube_radius=0.025, colormap='Vega20')
    mlab.plot3d(PhaseX, NewDop-0.35, np.zeros(len(phase)),   np.zeros(len(phase)), tube_radius=0.025, colormap='prism')


    PhaseDifference = np.abs(phase+Shift-min(phase)-PhaseCalc)

    i = int(np.where(PhaseDifference == np.min(PhaseDifference))[0])

    PointX = PhaseX[i-25:i+25]
    PointY = PhaseY[i-25:i+25]

    mlab.plot3d(PointX, PointY, np.zeros(len(PointX)),   np.ones(len(PointX)), tube_radius=0.12, colormap='Vega10')

    #Create same length time text
    TimeText = str(np.round(PhaseCalc+Shift,2))
    if len(TimeText)==3:
        TimeText+="0"
    TimeText=" Phase: "+TimeText
    mlab.text3d(5.0, 10.0, 0.0, TimeText, scale=(0.5, 0.5, 0.5))


    #the labels in the diagram
    mlab.text3d(-1.5, 1.0, 0.0, 'Phase', scale=(0.5, 0.5, 0.5))
    mlab.text3d(-7.0, 3.0, 0, 'Flux',  orientation=(0., 0., 90.), scale=(0.5, 0.5, 0.5))
    mlab.view(azimuth=0, elevation=0.0, distance=25, focalpoint=(0,7,0))
    return mlab.screenshot(antialiased=True)


mlab.figure(1, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), size=(1200, 800))
animation = mpy.VideoClip(make_frame, duration=TDur)

#Generating video and the animation
animation.write_gif("PhaseCurve.gif", fps=FPS)
animation.write_videofile("PhaseCurve.mp4", fps=FPS)
mlab.close()
