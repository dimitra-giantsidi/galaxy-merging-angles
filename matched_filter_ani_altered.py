#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:33:53 2022

@author: Ani
Note- this is directly converted from an IDL routine that Burçin Mutlu-Pakdil 
      had provided to me on 04/22/22

Note- bug fixes by Ben in 2025-26
"""
import os
import numpy
# import random
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve
# from astropy.io import fits
from astropy.table import Table
from astropy.io import fits
from astropy import stats

def get_xposvector(xarcmin_min, xarcmin_max, pixsize):
    """
    ;pixsize should be input in arcmin
    """
    print("x min ", xarcmin_min)
    print("x max ", xarcmin_max)
    xarray_out = numpy.arange(xarcmin_min+0.5*pixsize, xarcmin_max, pixsize)
    return xarray_out

def get_yposvector(xarcmin_min, xarcmin_max, pixsize):
    """
    ;pixsize should be input in arcmin
    """
    print("y min ", xarcmin_min)
    print("y max ", xarcmin_max)
    xarray_out = numpy.arange(xarcmin_min+0.5*pixsize, xarcmin_max, pixsize)
    return xarray_out

def mk_norm_backcmd(field, catalog, rmaglim=26.0, plot=False, \
                    RAkey = 'c1', DECkey = 'c2', gmagkey = 'gmag', \
                    # egmagkey = 'err_gmag', 
                    rmagkey = 'rmag', \
                    # ermagkey = 'err_rmag', \
                    agmagkey = 'Ag', armagkey = 'Ar',pix=90.):
    # pix=pix # arcsec
    pixelsize=pix#/60.0 #arcmin
    # pixelsize = pixelsize
    cat = Table.read(catalog, format='fits')
    RAin = cat[RAkey]
    DECin = cat[DECkey]
    gmagin = cat[gmagkey]
    # egmagin = cat[egmagkey]
    rmagin = cat[rmagkey]
    # ermagin = cat[ermagkey]
    agin = cat[agmagkey]
    arin = cat[armagkey]

    gmag = gmagin-agin
    rmag = rmagin-arin

    #Background region is defined here; may need to tweak a little
    midpoint_ra =  (min(RAin)+max(RAin))/2.  #define a background center
    midpoint_dec = (min(DECin)+max(DECin))/2. #define a background center
    bckrad= 120.0 #define a radius outside which is the background (needs to be same units as pix)

    Xposb = 60.0*(RAin-midpoint_ra)*numpy.cos(midpoint_dec/180.0*numpy.pi)
    Yposb = 60.0*(DECin-midpoint_dec)
    # Xposb = RAin
    # Yposb = DECin
    dist = numpy.sqrt(Xposb*Xposb+Yposb*Yposb)
    index = (dist > bckrad)

    g_fin = gmag[index]
    r_fin = rmag[index]
    gminr_fin = g_fin-r_fin
    back_num=len(r_fin)
  
    # DIDNT USED TO HAVE THE FIRST TERM -> BUGGGG
    areaback = (600*600) - (numpy.pi*bckrad*bckrad) # square field 10 deg edge length - central cut ->  for bg area
    areapixel = pixelsize*pixelsize
    arearatio = areaback/areapixel
  
    rbin = 0.15
    grbin = 0.15
    grmin=-0.5
    grmax=1.5
    rmin=18.0001
    rmax=rmaglim
  
    backstars, xedges, yedges = numpy.histogram2d(gminr_fin,r_fin,bins = [numpy.arange(grmin, grmax, grbin), \
                                                          numpy.arange(rmin, rmax, rbin)], \
                                                  range = [[grmin, grmax], [rmin, rmax]])
    
    backstars = backstars.T
    backstars = backstars/arearatio
    print('Total number of stars in background CMD')
    print(numpy.sum(backstars))

    if plot:
    
        fig, axarr = plt.subplots(1, 2, figsize = (10, 5))
        axarr[0].scatter(gminr_fin, r_fin, s = 1)
        axarr[0].set_xlim([grmin, grmax])
        axarr[0].set_ylim([rmin, rmax])
        axarr[0].set_xlabel(r'$g-r$')
        axarr[0].set_ylabel(r'$r$')
        axarr[0].invert_yaxis()
        
        axarr[1].imshow(backstars[::-1, :], extent=[xedges[0], xedges[-1], \
                                                    yedges[0], yedges[-1]], \
                                            aspect="auto")
        axarr[1].set_xlim([grmin, grmax])
        axarr[1].set_ylim([rmin, rmax])
        axarr[1].set_xlabel(r'$g-r$')
        axarr[1].set_ylabel(r'$r$')
        axarr[1].invert_yaxis()
        plt.show()
        # plt.savefig('../plots/storm_4096_3/' + field + '_backmap.png', bbox_inches='tight')
        # plt.close()
    return backstars
    
def mk_norm_objcmd_gd(field, sourcefile, rmaglim = 26.0, plot=False):
    '''
    This is the signal file. You can create a signal file by populating a
    stellar population from an isochrone and its luminosity function.
    This is basically the cmd space which the code will try to match.
    '''
    
    sourcefile = Table.read(sourcefile, format='ascii')
    gminr_mock = sourcefile['grcol']
    r_mock = sourcefile['rmag']
    
    rbin = 0.15
    grbin = 0.15
    grmin=-0.5
    grmax=1.5
    rmin=18.0001
    rmax=rmaglim
    
    stars, xedges, yedges = numpy.histogram2d(gminr_mock, r_mock, bins = [numpy.arange(grmin, grmax, grbin), \
                                                          numpy.arange(rmin, rmax, rbin)], \
                                                  range = [[grmin, grmax], [rmin, rmax]])
    stars = stars.T
    stars = stars/numpy.sum(stars)
    print('Total number of signal CMD')
    print(numpy.sum(stars))
    
    if plot:
        fig, axarr = plt.subplots(1, 2, figsize = (10, 5))
        axarr[0].scatter(gminr_mock, r_mock, s = 1)
        axarr[0].set_xlim([grmin, grmax])
        axarr[0].set_ylim([rmin, rmax])
        axarr[0].set_xlabel(r'$g-r$')
        axarr[0].set_ylabel(r'$r$')
        axarr[0].invert_yaxis()
        
        axarr[1].imshow(stars[::-1, :], extent=[xedges[0], xedges[-1], \
                                                yedges[0], yedges[-1]], aspect="auto")
        axarr[1].set_xlim([grmin, grmax])
        axarr[1].set_ylim([rmin, rmax])
        axarr[1].set_xlabel(r'$g-r$')
        axarr[1].set_ylabel(r'$r$')
        axarr[1].invert_yaxis()
        plt.show()
        # plt.savefig('../plots/storm_4096_3/' + field + '_sigmalmap.png', bbox_inches='tight')
        # plt.close()
    
    return stars
  
def matched_filter_ani(objcmd, backcmd, my_save_path, plotdir, field, catalog, rmaglim=26.0, plot=False, \
                    RAkey = 'c1', DECkey = 'c2', gmagkey = 'gmag', \
                    # egmagkey = 'err_gmag', 
                    rmagkey = 'rmag', \
                    # ermagkey = 'err_rmag', 
                    agmagkey = 'Ag', armagkey = 'Ar',pix=90.,pdist=60.):

    # pix=pix #arcsec
    pixelsize=pix#/60.0 #in arcmin
    # pixelsize = pixelsize # in kpc now



    yo = (backcmd < 1.0e-7) #Bad indices?
    backcmd[yo] = numpy.mean(backcmd)/10.0

    wghtcmd = objcmd/backcmd

    magbin = 0.15
    termtwo = magbin*magbin*numpy.sum(objcmd)
    denom = magbin*magbin*numpy.sum(wghtcmd*objcmd)

    #extiction correction for g, r-magnitude, r-magnitude error, etc
    tab = Table.read(catalog, format='fits')
    RA = tab[RAkey]
    DEC = tab[DECkey]
    gmag = tab[gmagkey]
    # egmag = tab[egmagkey]
    rmag = tab[rmagkey]
    # ermag = tab[ermagkey]
    ag = tab[agmagkey]
    ar = tab[armagkey]

    g_fin = gmag-ag #Correct for extinction
    r_fin = rmag-ar 
    gminr_fin = g_fin-r_fin

    #Center of the object you are exploring or the center of the image
    RA_0 = 0 #<< in degrees
    DEC_0= 0 #<< in degrees
    pdist = pdist
    #get 'distance' of each point.
    #define ra and dec range to plot -- in arcmin

    # yrange = [60.0*min(DEC-DEC_0), 60.0*max(DEC-DEC_0)]
    yrange = [-pdist,pdist] # in arcmin 
    # xrange = [60.0*min((RA-RA_0)*numpy.cos(numpy.pi*DEC_0/180.0)), \
    #           60.0*max((RA-RA_0)*numpy.cos(numpy.pi*DEC_0/180.0))]
    xrange = [-pdist,pdist]

    # Xpos = 60.0*(RA-RA_0)*numpy.cos(DEC_0/180.0*numpy.pi)
    # Ypos = 60.0*(DEC-DEC_0)
    Xpos = 60.*(RA-RA_0)*numpy.cos(DEC_0/180.0*numpy.pi) # deg to arcmin
    Ypos = 60.*DEC-DEC_0

    xvec_pos = get_xposvector(xrange[0],xrange[1],pixelsize)
    yvec_pos = get_yposvector(yrange[0],yrange[1],pixelsize)
    xels = len(xvec_pos)
    yels = len(yvec_pos)

    mapp = numpy.zeros((xels,yels))

    #You can play with bin sizes and cmd ranges you want to focus.
    rbin = 0.15
    grbin = 0.15
    grmin=-0.5
    grmax=1.5
    rmin=18.0001
    rmax=rmaglim

    for i in range(len(xvec_pos)):
        for j in range(len(yvec_pos)):
            xcurrmax = xvec_pos[i]+0.5*pixelsize
            xcurrmin = xvec_pos[i]-0.5*pixelsize
            ycurrmax = yvec_pos[j]+0.5*pixelsize
            ycurrmin = yvec_pos[j]-0.5*pixelsize 

            goodindex = ((Xpos < xcurrmax) & (Xpos > xcurrmin) & \
                        (Ypos < ycurrmax) & (Ypos > ycurrmin))
                
            goodcount = len(Xpos[goodindex])
            if goodcount == 0:
                termone = 0.0
            else:
                r_map = r_fin[goodindex]
                gminr_map = gminr_fin[goodindex]
                cmd_at_pixel, xedges, yedges = numpy.histogram2d(gminr_map,r_map, bins = [numpy.arange(grmin, grmax, grbin), \
                                                                         numpy.arange(rmin, rmax, rbin)], \
                                                 range = [[grmin, grmax], [rmin, rmax]])
                cmd_at_pixel = cmd_at_pixel.T
                termone = numpy.sum(wghtcmd*cmd_at_pixel)*magbin*magbin
    
            mapp[i,j] = (termone-termtwo)/denom
    
    #Note- I replaced the smoothing function (create_exp_fitler.pro & filter_image_beth.pro above)
    #with the astropy 2D gaussian smoother; I think this is OK - AC 06/21/22
    kernel = Gaussian2DKernel(x_stddev=0.7) 
    smoothmap = convolve(mapp, kernel)

    #Note- I replaced the mean, median, mode, sigma estimation (mmm.pro) with the astropy
    #sigma clip function to estimate the sky background & fluctuation; I think 
    #this is OK, and there is a diagnostic plot that is outputted to check - AC 06/21/22
    mean, median, sigma = stats.sigma_clipped_stats(smoothmap, sigma=2, maxiters=4)
    # setting edges to median - dumb fix but works
    # smoothmap[0, :] = median
    # smoothmap[-1, :] = median
    # smoothmap[:, 0] = median
    # smoothmap[:, -1] = median

    mean_nosm, median_nosm, sigma_nosm = stats.sigma_clipped_stats(smoothmap, sigma=2, maxiters=4)
    clipped_data = stats.sigma_clip(smoothmap, sigma=2, maxiters=4)
    
    if plot:
        plt.figure()
        plt.hist(numpy.ravel(clipped_data[clipped_data.mask]), label = 'Object')
        plt.hist(numpy.ravel(clipped_data[~clipped_data.mask]), label = 'Sky')
        plt.gca().set_yscale('log')
        plt.ylabel('N', fontsize = 15)
        plt.xlabel('Counts', fontsize = 15)
        plt.legend(fontsize = 15)
        plt.show()
        plt.close()
    # plt.savefig('../plots/storm_4096_3/' + field + '_background_object.png', bbox_inches='tight')
    # plt.close()

    
    print('Sigma in smoothed stellar counts ' + str(sigma))
    print('Mean in smoothed stellar counts ' + str(mean))

    # ;You can use different contour levels here...
    levels = [4,5,7,10,20,50,100,150]
    
    # ; Convert the smoothed number density array to units of "sigma" for
    #   ease of plotted the figure
    sig_array = (smoothmap-median)/sigma
    sig_array = sig_array.T
    sig_array_under = numpy.flip(sig_array,axis=1)
    
    if not os.path.exists(plotdir):
            os.makedirs(plotdir)

    mapplot=plotdir+'mf_'+str(field)+'.png'


    plt.figure()
    plt.imshow(sig_array[::-1, :], extent=[xrange[0], xrange[1],  yrange[0], yrange[1]], \
            cmap = 'gray_r', vmin = -15, vmax = 30)
    # plt.contour(xvec_pos, yvec_pos, sig_array, levels=levels)
    # plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta$x (arcmin)')
    plt.ylabel(r'$\Delta$y (arcmin)')
    # plt.xlim(-pdist,pdist)
    # plt.ylim(-pdist,pdist)


    hdu = fits.PrimaryHDU(sig_array)
    hdu.header['SRC_ID'] = my_save_path
    hdu.writeto(my_save_path, overwrite=True)
    
    plt.show()
    plt.savefig(mapplot, bbox_inches = 'tight')
    plt.close()

