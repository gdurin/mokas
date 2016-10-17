# polar is a collection of routines 
# to handle various plot and calculation on a polar coordinate system

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def to_radians(theta):
	return theta/180*np.pi

def to_degree(theta):
	return theta*180/np.pi

def r_ellipse_center_on_focus(theta,a,b):
	if a >= b:
		ecc = (1-(b/a)**2)**0.5
		return (b*b/a)/(1-np.cos(theta)*ecc)
	else:
		ecc = (1-(a/b)**2)**0.5
		return (b*b/a)/(1-np.cos(theta)*ecc)

def cart2polar(xy,origin=(0,0),swope_xy=False, reverse_y=True,ordered=True):
	"""
	transform cartesian coordinates into polar ones
	Angle is in radiants
	"""
	if swope_xy:
		yc,xc = origin
		x,y = (xy[:,1] - xc), (xy[:,0] -yc)
	else:
		xc,yc = origin
		x, y = (xy[:,0] - xc), (xy[:,1] -yc)
	theta = np.arctan2(y,x)
	if reverse_y:
		theta = -theta
	r = (x*x + y*y)**0.5
	if ordered:
		order = np.argsort(theta)
		theta = theta[order]
		r = r[order]
	return r, theta


def plot_rays(center,step_angle,ax,axis_limits=None):
        xc,yc = center
        if axis_limits is None:
        	axis_limits = ax.axis()
        xmin, xmax, ymax, ymin = axis_limits
        x = np.linspace(xmin,xmax,50)
        for m in np.arange(0,np.pi/2,step_angle):
            for sign in [-1,1]:
            	y = sign*np.tan(m)*(x-xc)+yc
    	        selection = (y>=ymin) & (y<=ymax)
    	        ax.plot(x[selection],y[selection],'k--',lw=0.5)
        ax.plot((xc,xc),(ymin,ymax),'k--',lw=0.5)

def calc_velocity(contours,origin,n_new_thetas=720,swope_xy=True):
	"""
	Calculated the velocity given a set of contours, an origin and the angles 

	Parameters:
	===========
	contours : dict
		The set of contours
	origin : tuple
		origin of the x-y axis
	n_new_thetas : n. of angles uniformly spaced along 360 degrees
	"""
	switches = sorted(contours.keys())
	delta_t = 1.
	new_thetas = np.linspace(-np.pi,np.pi,n_new_thetas)
	for switch in switches:
		r, theta = cart2polar(contours[switch],origin,swope_xy=swope_xy)
		r_inter = np.interp(new_thetas,theta,r)
		if switch == switches[0]:
			r0 = r_inter
		else:
			v_ist = (r_inter - r0) / delta_t
			r0 = r_inter
			if switch == switches[1]:
				v = v_ist
			else:
				v = np.vstack((v,v_ist))
	times = (switches[1:] - switches[1]) / delta_t
	v = pd.DataFrame(v,columns=new_thetas,index=times)
	return v

def calc_mean_error_velocity(v):
	#v_mean = pd.rolling_mean(v,10)
	time_steps = v.shape[0]
	v_mean = v.mean(0)
	v_err = v.std(0)/(time_steps)**0.5
	return v_mean, v_err

def plot_mean_velocity(x,v_mean,v_err,fig=None,ax=None,title=None,label=None,color='k'):
	"""
	Plot the velocity for different angles
	"""
	if fig is None:
		fig, ax = plt.subplots(1,1)
	# v is a pandas.DataFrame, with rows as time, and cols as thetas
	#ax.plot(v.columns*180/np.pi,v_mean,'o')
	ax.errorbar(to_degree(x),v_mean,v_err,fmt='o',c=color,label=label)
	ax.grid(True)
	ax.set_xlabel("angle (deg)")
	ax.set_xticks(np.arange(-4,5)*45)
	ax.set_ylabel("average velocity")
	ax.set_title(title)
	return


def plot_displacement(contours,origin,reference='nucleated_domain',
					n_new_thetas=720,swope_xy=True,
					fig=None,ax=None,title=None,step_in_frames=10):
	"""
	Plot dispacements at different angles

	Parameters:
	===========
	contours : dict
		A dict containing the contours at different switches
	origin : tuple
		center of the axis
	reference : string
		how to calculate the displacement
		center : from the center of mass of the nucleated domain
		nucleated_domain : from the nucleated domain contour
		differential : from the previous contour
	swope_xy : bool
		swope x-y axis
	"""
	if fig is None:
		fig, ax = plt.subplots(2,1,sharex=True)
	switches = sorted(contours.keys())

	syb = '-'
	if reference == 'differential':
		syb = 'o'
	new_thetas = np.linspace(-np.pi,np.pi,n_new_thetas)
	for switch in switches:
		r, theta = cart2polar(contours[switch],origin,swope_xy=swope_xy)
		frame = switch - switches[0]
		lw = 1.5*(frame%step_in_frames==0) + 0.5
		if reference is not 'center':
			new_r = np.interp(new_thetas,theta,r)
			if switch == 0:
				r0 = new_r
			delta_r = new_r-r0
			ax.plot(to_degree(new_thetas), delta_r, 'k'+syb,lw=lw)
			if reference == 'differential':
				r0 = new_r
		else:
			ax.plot(to_degree(theta), r, 'k'+syb,lw=lw)


	ax.grid(True)
	ax.set_xlabel("angle (deg)")
	ax.set_xticks(np.arange(-4,5)*45)
	if reference=='nucleated_domain':
		ax.set_ylabel("distance from the nucleated domain")
	else:
		ax.set_ylabel("distance from the center")
	ax.set_title(title)
	# return the last contour
	# and the n. of frames between the first and the last switches
	frames = switches[-1] - switches[1] + 1
	if reference is not 'center':
		return new_thetas, delta_r, frames
	else:
		return theta, r, frames

if __name__ == "__main__":
	Bxs = [-200,-100,-50,0,50,100,200]
	plt.close("all")
	fig1, (ax1) = plt.subplots(1,1)
	fig2, (ax2,ax3) = plt.subplots(2,1,sharex=True)
	# theta = np.linspace(0,2*np.pi,100)
	# r = r_ellipse_center_on_focus(theta,a,b)
	# x = r * np.cos(theta+np.pi/2)
	# y = r * np.sin(theta+np.pi/2)
	# ax.plot(theta,r,'-')
	with open("all_contours.pkl",'rb') as f:
		all_contours = pickle.load(f)
	with open("all_centers.pkl",'rb') as f:
		all_centers = pickle.load(f)
	Bx = -100
	cnt = all_contours[Bx]
	origin = all_centers[Bx]
	for k,key in enumerate(sorted(cnt.keys())):
		xy = cnt[key]
		lw = 1.5*(k%10==0) + 0.5
		ax1.plot(xy[:,1,]-origin[1], xy[:,0]-origin[0],'k-',lw=lw)
	ax1.set_aspect('equal')
	ax1.invert_yaxis()
	ax1.grid(True)
	plot_displacement(cnt,origin,reference='center',swope_xy=True,fig=fig2,ax=ax2)
	plot_displacement(cnt,origin,reference='nucleated_domain',swope_xy=True,fig=fig2,ax=ax3)
	fig2.suptitle(r"$B_x$ = %i mT" % Bx,fontsize=30)
	#
	plot_mean_velocity(cnt,origin,n_new_thetas=37,swope_xy=True)
	plt.show()