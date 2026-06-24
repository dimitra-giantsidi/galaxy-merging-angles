import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
import vaex
from photerr import LsstErrorModel
import glob 
from astropy import stats
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord, CartesianRepresentation
from astropy.io import ascii, fits
from astropy.table import Table, vstack
import astropy.units as u

def get_xposvector(xarcmin_min, xarcmin_max, pixsize):
    """
    ;pixsize should be input in arcmin
    """
    print("x min ", xarcmin_min)
    print("x max ", xarcmin_max)
    xarray_out = np.arange(xarcmin_min+0.5*pixsize, xarcmin_max, pixsize)
    return xarray_out

def get_yposvector(xarcmin_min, xarcmin_max, pixsize):
    """
    ;pixsize should be input in arcmin
    """
    print("y min ", xarcmin_min)
    print("y max ", xarcmin_max)
    xarray_out = np.arange(xarcmin_min+0.5*pixsize, xarcmin_max, pixsize)
    return xarray_out

# spherical to cartesian in units of D
def spheres(data_coords,c,D):
    """
    data_coords: coordinates of your data as a SkyCoords object
    c: the center point that you want to define your physical coordinate system around -> the "zero" of your axes
    """
    seps = data_coords.separation(c)
    poss = data_coords.position_angle(c)
    x = seps.degree*np.sin(-1.0*poss.radian)*(np.pi/180)*D
    y = seps.degree*np.cos(poss.radian)*(np.pi/180)*D
    return x, y

# generates error files for signal files
def err_file_gen(df,name,magkey,errkey,ret=False):
    """
    generates files for photometric errors in signal file generation
    """
    bins = np.arange(16, 29, 0.15)
    bincenters = (bins[:-1] + bins[1:])/2
    err_per_bin = np.empty(bins.size - 1)
    for i in range(bins.size - 1):
        mask = ((df[magkey] >= bins[i]) & (df[magkey] <= bins[i+1]))
        err_per_bin[i] = np.mean(df[errkey].loc[mask])

    bin1 = magkey+'1'
    bin2 = magkey+'2'
    mag_err_table = Table()
    mag_err_table[bin1] = bins[:-1]
    mag_err_table[bin2] = bins[1:]
    mag_err_table['err'] = err_per_bin

    dir = f"../../data/matched-filter/iso_lf_signals/{name}_errcomp/"
    if not os.path.exists(dir):
        os.makedirs(dir)

    outstr = dir + f"{name}_{magkey}errs.txt"

    ascii.write(mag_err_table,outstr,overwrite=True)
    if ret:
        return mag_err_table
    else:
        return

# generates signal files from isochrones, lf, error
def signal_gen(name, num, box, age, mh, D, year, plot=True):
    """
    Generates a signal file with lsst errors for mock observed ananke star particles
    name: data table name, used for reading / writing 
    age: stellar pop age 
    mh: stellar pop metallicity 
    D: distance, in pc, for dmod
    """
    # amke sure path has proper box
    # path = f"/Users/f0080bw/Desktop/scialog/ananke_testing/{box}_catalogs_plots/" + name + "/survey." + name + ".0.h5"# for local 
    path = f"../../data/raw/{box}_{num}a/survey.{name}.0.h5"
    vframe = vaex.open(path)
    dframe = pd.DataFrame(vframe,columns=vframe.column_names)
    errModel = LsstErrorModel(nYrObs=year)
    errModel = LsstErrorModel(renameDict={ "u": "lsst_u", "g": "lsst_g", "r": "lsst_r", "i": "lsst_i", "z": "lsst_z", "y": "lsst_y"}) 
    dframe_w_errors = errModel(dframe)

    # doing g and r since those are the bg data we have for now
    err_file_gen(dframe_w_errors,name=name,magkey='lsst_g',errkey='lsst_g_err')
    err_file_gen(dframe_w_errors,name=name,magkey='lsst_r',errkey='lsst_r_err')

    # PARSEC:
    iso = ascii.read(f"../../data/matched-filter/signals/parsec_{age}{mh}_iso.txt")
    order = np.argsort(iso['rmag'])
    mag2 = iso['rmag'][order]
    mag1 = iso['gmag'][order]

    # PARSEC:
    lf = ascii.read(f"../../data/matched-filter/iso_lf_signals/iso_lf/parsec_{age}{mh}_lf.txt")
    mag2_bins = lf['magbinc']
    mag2_counts = lf['rmag'] * 1e7 # can pick any mass here - signal CMD is a density, so this will populate the whole isochrone with no low number stats issues

    # reading in error per bin from the LSST err model 
    mag2err = ascii.read("../iso_lf_signals/" + name + "_errcomp/"+ name + "_lsst_rerrs.txt")
    mag1err = ascii.read("../iso_lf_signals/" + name + "_errcomp/"+ name + "_lsst_gerrs.txt")

    r1 = mag2err['lsst_r1']
    r2 = mag2err['lsst_r2']
    err_r = mag2err['err']

    g1 = mag1err['lsst_g1']
    g2 = mag1err['lsst_g2']
    err_g = mag1err['err']

    # setting nans to a minimum value for the error per bin
    err_r = np.nan_to_num(err_r, nan=0.01, posinf=0.01, neginf=0.01)
    err_g = np.nan_to_num(err_g, nan=0.01, posinf=0.01, neginf=0.01)

    r_centers = 0.5 * (r1 + r2)
    g_centers = 0.5 * (g1 + g2)

    # interpolating over errs per bin for faster assignment of error per source, instead of doing it star by star bin per bin
    r_err_interp = lambda mags: np.interp(mags, r_centers, err_r)
    g_err_interp = lambda mags: np.interp(mags, g_centers, err_g)

    np.random.seed(0)
    syn_stars_mag1, syn_stars_mag2 = [],[]
    bin_size = 0.15 # amke sure this matches what was set when generating lf from website 

    for bin, lam in zip(mag2_bins, mag2_counts):
        # sampling from poisson, bc counts per bin
        num = np.random.poisson(lam=lam)
        if num == 0:
            continue

        mag2_samples = np.random.uniform(bin - bin_size/2, bin + bin_size/2, size=num)
        
        # Interpolate mag1 from isochrone
        mag1_samples = np.interp(mag2_samples, mag2, mag1)

        # Apply photometric errors to synthetic stars taken from lf + isochrone
        # errors for each from lambda functions 
        r_errs = r_err_interp(mag2_samples)
        g_errs = g_err_interp(mag1_samples)

        #shifting them by an ammount given by the error per bin
        mag1_samples += np.random.normal(0, g_errs)
        mag2_samples += np.random.normal(0, r_errs)
        
        syn_stars_mag1.extend(mag1_samples)
        syn_stars_mag2.extend(mag2_samples)

    syn_stars_mag1 = np.array(syn_stars_mag1)
    syn_stars_mag2 = np.array(syn_stars_mag2)

    dmod = 5*np.log10(D) - 5 # D in parsec!!! - kpc in filename
    tab = Table([syn_stars_mag1 - syn_stars_mag2, syn_stars_mag2 + dmod],names=('grcol','rmag')) # change these depending on mags used 

    ascii.write(tab,f'../../data/matched-filter/iso_lf_signals/signal_files/parsec_{int(D/1000)}_{age}{mh}_signal_cmd_gr.txt',overwrite=True)

    if plot: 
        plt.figure(figsize=(6, 8))
        # Plot input isochrone
        # plt.plot(
        #     mag1 - mag2,
        #     mag2 + dmod,
        #     c="red", lw=1.5, label="Isochrone"
        # )

        # Plot synthetic stars that ahve been shifted
        plt.scatter(
            syn_stars_mag1 - syn_stars_mag2,
            syn_stars_mag2 + dmod,
            s=10, c="black", alpha=1, label="Synthetic stars"
        )

        plt.gca().invert_yaxis()
        plt.ylim(27,21)
        plt.xlabel("g - r (mag)")
        plt.ylabel("r (mag)")
        plt.title("Synthetic CMD with Photometric Errors")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        # plt.savefig("/home/bvelguth/scialog/matched_filter_testing/plots/cmd.png")
    return

# takes in bg catalog, calculates density as a func of color or mag - useful for plotting
def bgmag_ndens(bgcat,mag,binsize,fov):
    if mag == 'g' or mag == 'r':
        data = bgcat[mag].value
    elif mag == 'g - r':
        data = bgcat['g'] - bgcat['r']
    
    bins = np.arange(data.min(), data.max() + binsize, binsize)
    bins_centers = (bins[:-1] + bins[1:]) / 2
    dens = []
    for i in range(len(bins) - 1):
        if i == 0:
            mask = np.where((data >= bins[i]) & 
                            (data <= bins[i+1]))[0]
        else:
            mask = np.where((data > bins[i]) & 
                            (data <= bins[i+1]))[0]
        dens.append(len(data[mask])/fov)
    return bins_centers, np.array(dens)

# function for background catalog creation
def bg_gen(bgpaths,mlim,D,binsize=0.01,edgelength=5,plot=True,save=True):
    """
    reads in a set of megacam files, scales up bg sources to area of my choice while keeping surface density the same 
    """
    # how many fields are being combined here, multiplied by one megacam fov in deg
    # need to do everything in angular units
    numfield = len(bgpaths)
    print(numfield)
    print(bgpaths)
    # area of observed bg in angluar area 
    bgfov = ((24/60)**2) * numfield
    # combines fields to one big table
    catlst = []
    for path in bgpaths:
        temp = Table(ascii.read(path))
        catlst.append(temp)
    bgcat = vstack(catlst)

    bgcat['g'] = bgcat['gmag'] - bgcat['Ag']
    bgcat['r'] = bgcat['rmag'] - bgcat['Ar']
    # takes in bg catalog, calculates spacial density as a func of color or mag - useful for plotting 
    colbins_centers, coldens = bgmag_ndens(bgcat=bgcat, mag = 'g - r', binsize=binsize, fov=bgfov)
    gmag_bins_centers, gdens = bgmag_ndens(bgcat=bgcat, mag='g',binsize=binsize,fov=bgfov)
    rmag_bins_centers, rdens = bgmag_ndens(bgcat=bgcat, mag='r',binsize=binsize,fov=bgfov)

    if plot:
        # diagnostic plots showing CMD and counts / area as a func of mag or color
        plt.figure()
        plt.scatter(bgcat['g'] - bgcat['r'],bgcat['r'],s=1,alpha=0.1)
        plt.xlabel('g-r')
        plt.ylabel('r')
        plt.gca().invert_yaxis()
        plt.show()

        plt.figure()
        plt.scatter(colbins_centers, coldens, s=5, label='g - r')
        plt.xlabel('g-r')
        plt.ylabel('counts/area')
        plt.show()

        plt.figure()
        plt.scatter(gmag_bins_centers, gdens, s=5, label='g')
        plt.scatter(rmag_bins_centers, rdens, s=5, label='r')
        plt.gca().axvline(x=25,linestyle='--',c='r',label='~YR1 Depth')
        plt.xlabel('mag')
        plt.ylabel('Counts / area')
        plt.legend(loc='upper left');
        plt.show()
    # new area to scale up to, keeping density the same 
    newarea = edgelength**2
    # number of total counts needed to fill newarea with same density 
    goal_counts = int(np.sum(gdens) * newarea) - len(bgcat) # subtracting actual num, so I only generate as many as necessary
    # 2d histogram of bg sources in gr vs r
    hist, *edges = np.histogram2d(bgcat['r'],(bgcat['g'] - bgcat['r']),bins=(gmag_bins_centers.size,colbins_centers.size))
    # centers of the 2d bins 
    centers = [(edges[0][:-1] + edges[0][1:]) / 2, (edges[1][:-1] + edges[1][1:]) / 2]
    # converting each bin to a density, counts/bin divided by total counts
    hist_dens = hist / len(bgcat)
    print(f"Goal: {goal_counts}")
    
    if plot:
        plt.figure()
        plt.imshow(hist_dens)
        plt.show()
    # scaling up by goal counts - dens per bin times goal counts gives new counts per bin
    hist_tosamp = hist_dens * goal_counts # each bin in col vs mag has a number to sample
    # drawing number of points per cmd bin as detemined by hist_tosamp  
    cols_generated, rmags_generated = [],[]
    for i in range(hist_tosamp.shape[0]): # indicies for magnitudes
        for j in range(hist_tosamp.shape[1]): # indicies for colors 
            # poisson draw for each bin - num per bin
            num_to_samp = int(np.random.poisson(lam=hist_tosamp[i,j]))
            # centers is arranged (mag centers,color centers) - same as the histogram, 2dhistogram is stupid
            # defining edges of each color and mag bin
            magbin_edges = (centers[0][i] - (binsize/2), centers[0][i] + (binsize/2))
            colbin_edges = (centers[1][j] - (binsize/2), centers[1][j] + (binsize/2))
            # these are arrays of colors/magnitudes that have been drawn from each bin according to the number of expected stars in that bin
            magdraw = np.random.uniform(magbin_edges[0],magbin_edges[1],num_to_samp)
            coldraw = np.random.uniform(colbin_edges[0],colbin_edges[1],num_to_samp)
            cols_generated.extend(coldraw)
            rmags_generated.extend(magdraw)

    rmags_generated = np.array(rmags_generated)
    cols_generated = np.array(cols_generated)
    # check: are generatied counts close to goal? poisson will make it vary on the 1% level
    print(f"Actual Counts: {len(cols_generated)}, Goal: {goal_counts}") 

    if plot:
        plt.figure()
        plt.scatter(cols_generated,rmags_generated,s=0.1,alpha=0.01)
        plt.gca().invert_yaxis()
        plt.title('Synthetic g-r and r drawn from 2D histogram')
        plt.show()
    
    # arrays of len goalcounts + len bgcat -> using the actual measurements as part of the synthetic bg
    gr_realsyn = np.concatenate([bgcat['gmag'].value - bgcat['rmag'].value,cols_generated])
    r_realsyn = np.concatenate([bgcat['rmag'].value,rmags_generated])
    g_generated = cols_generated + rmags_generated # getting g from color and r
    g_realsyn = np.concatenate([bgcat['gmag'].value,g_generated])

    if plot:
        # testing new synthetic + real catalogs for density rmag -> color and gmag works too i just didnt plot it
        bins_realsyn = np.arange(r_realsyn.min(), r_realsyn.max() + binsize, binsize)
        bins_centers_realsyn = (bins_realsyn[:-1] + bins_realsyn[1:]) / 2
        dens_realsyn = []
        for i in range(len(bins_realsyn) - 1):
            if i == 0:
                mask = np.where((r_realsyn >= bins_realsyn[i]) & 
                                (r_realsyn <= bins_realsyn[i+1]))[0]
            else:
                mask = np.where((r_realsyn > bins_realsyn[i]) & 
                                (r_realsyn <= bins_realsyn[i+1]))[0]
            dens_realsyn.append(len(r_realsyn[mask])/(newarea))
        plt.figure()
        plt.scatter(bins_centers_realsyn,dens_realsyn,label='Obs + Drawn counts / area')
        plt.scatter(rmag_bins_centers, rdens, s=5, label='Obs only / area')
        plt.legend()
        plt.show()

    # random gneeration of locations for these points 
    n_points = len(g_realsyn) 
    points = np.empty((n_points,2))
    points[:,0] = np.random.uniform(-edgelength/2, edgelength/2,n_points)
    points[:,1] = np.random.uniform(-edgelength/2, edgelength/2,n_points)

    # drawing extionction values from dist of obs one for the synthetic bg points
    Ag_syn = np.random.normal(np.mean(bgcat['Ag']),np.std(bgcat['Ag'],ddof=1),size=len(g_generated))
    Ar_syn = np.random.normal(np.mean(bgcat['Ar']),np.std(bgcat['Ar'],ddof=1),size=len(rmags_generated))

    bg_for_mf = Table()
    bg_for_mf['gmag'] = g_realsyn # makes some of the values the measured ones, but randomly positioned
    bg_for_mf['rmag'] = r_realsyn
    bg_for_mf['c1'] = points[:,0] # in deg now again
    bg_for_mf['c2'] = points[:,1]
    bg_for_mf['Ag'] = np.concatenate([bgcat['Ag'].value, Ag_syn])
    bg_for_mf['Ar'] = np.concatenate([bgcat['Ar'].value, Ar_syn])

    # writing bg field - use later in combo function
    if save:
        outstr = f'../../data/matched-filter/bgcats/bgr{mlim}g{mlim}_for_mf_{int(edgelength)}deg.fits' 
        bg_for_mf.write(outstr,format='fits',overwrite=True)
        # ascii.write(bg_for_mf,outstr,overwrite=True) # bg, maglim, for mf, length of field
        return outstr
    return 

# function that finds best plot size for each galaxy - NOT USING
def plot_size(box,num,D):
    # want to read in the ananke output dataframe from ananke test and use it to find size of galaxy to plot
    name = f'{box}_4096_{num}_data_{D}_26p5' # only doing deepest image, makes plots at each dist the same size for comparison
    vframe = vaex.open(f"/Users/f0080bw/Desktop/scialog/ananke_testing/{box}_catalogs_plots/" + name + "/survey." + name + ".0.h5")
    dframe = pd.DataFrame(vframe,columns=vframe.column_names)
    
    # magnitudes of 3d position vectors of each star
    posvecmag = np.linalg.norm(np.array([dframe['px']-np.mean(dframe['px']),
                                         dframe['py']-np.mean(dframe['py']),
                                         dframe['pz']-np.mean(dframe['pz'])]),axis=0)
    # bin the above
    shells = np.linspace(posvecmag.min(),posvecmag.max(),1000)
    # total mass
    mtot = np.sum(dframe['mact'])
    # goes through range of distances
    for i in range(len(shells)):
        # defines sphere of R <= posvecmag.max()
        mask = np.where(posvecmag <= shells[i])
        # finds total mass of stars within that sphere
        m = np.sum(dframe['mact'].iloc[mask])

        # breaks when total mass contained within sphere is at or above some% of total 
        if m/mtot >= 0.995:
            size = shells[i]
            break
    
    pdist = (size/D) * (180 / np.pi)
    return pdist # in degrees

# combines mock obs from anake with bg field of choice 
def bg_dwarf_combo(pdist,name,year,D,c1,c2,plotdir,dwarfcatpath,bgcatpath,edgelength=5,plot=True,save=True):
    vdwarfcat = vaex.open(dwarfcatpath)
    dwarfcat = pd.DataFrame(vdwarfcat,columns=vdwarfcat.column_names)
    
    # adding errors to magnitudes - see how the cmd smears out at the dim magnitudes?
    errModel = LsstErrorModel(nYrObs=year)
    errModel = LsstErrorModel(renameDict={ "u": "lsst_u", "g": "lsst_g", "r": "lsst_r", "i": "lsst_i", "z": "lsst_z", "y": "lsst_y"}) 
    dframe_w_errors = errModel(dwarfcat, random_state=42)

    fig = plt.figure(figsize=(6,8))
    plt.scatter(dframe_w_errors.lsst_g - dframe_w_errors.lsst_r, dframe_w_errors.lsst_r,alpha=1,s=0.1,c=dframe_w_errors.age)
    plt.gca().invert_yaxis()
    plt.show()

    # these are xyz positions relative to observer
    c1dwarf = dframe_w_errors[c1] 
    c2dwarf = dframe_w_errors[c2] 
    # c3dwarf = dwarfcat[c3] 
    c = SkyCoord(x=c1dwarf,y=c2dwarf,z=D,unit='kpc',representation_type='cartesian')

    phi = c.spherical.lon.to(u.deg).value   # azimuth
    theta = c.spherical.lat.to(u.deg).value # latitude (90 - polar angle)

    mean_coord = SkyCoord(np.mean(phi)*u.deg, np.mean(theta)*u.deg, frame='icrs')
    offset_frame = mean_coord.skyoffset_frame()
    c_offset = c.transform_to(offset_frame)

    phi_corr = c_offset.lon.to(u.deg).value
    theta_corr = c_offset.lat.to(u.deg).value

    plt.figure(figsize=(10.5, 8))
    plt.scatter((theta_corr),-(phi_corr), s=1, c=dwarfcat.age.values)
    plt.colorbar(label=r'$log_{10} (age/yr)$')
    plt.xlabel('Azimuthal angle [deg]')
    plt.ylabel('Polar angle [deg]')
    plt.title('Angular projection centered on mean position')
    plt.axhline(0, color='gray', ls='--', lw=1)
    plt.axvline(0, color='gray', ls='--', lw=1)
    plt.xlim(-pdist,pdist)
    plt.ylim(-pdist,pdist)

    if save:
        savepath = f"../../data/plots/{plotdir}"
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.savefig(f"{savepath}/{name}_{c1}{c2}_spherical_proj_{int(np.round(pdist*60,0))}.png", bbox_inches = 'tight')

    if plot:
        plt.show()
    elif not plot:
        plt.close()
        
    # c1 and c2 are in DEGREES 
    bgcat = Table.read(bgcatpath,format='fits')
    c1bg = bgcat['c1']
    c2bg = bgcat['c2']
    # bgcoord = SkyCoord(bgcat['RA'],bgcat['DEC'],unit='deg',frame='icrs')
    # c1bg,c2bg = spheres(bgcoord,c=SkyCoord(0,0,unit='deg',frame='icrs'),D=D)

    bgtable = Table()
    bgtable['c1'] = c1bg # distances on axes in units of degrees
    bgtable['c2'] = c2bg
    bgtable['gmag'] = bgcat['gmag']
    bgtable['Ag'] = bgcat['Ag'] # no extinction for simulated stars - is this valid i think so 
    bgtable['rmag'] = bgcat['rmag']
    bgtable['Ar'] = bgcat['Ar']
    bgtable['flag'] = 'bg'
    bgtable['age'] = -99

    dwarftable = Table()
    dwarftable['c1'] = theta_corr # distances on axes in degrees
    dwarftable['c2'] = -phi_corr
    dwarftable['gmag'] = dframe_w_errors.lsst_g
    dwarftable['Ag'] = np.zeros(len(dwarfcat.lsst_g_Err)) # no extinction for simulated stars - is this valid i think so 
    dwarftable['rmag'] = dframe_w_errors.lsst_r
    dwarftable['Ar'] = np.zeros(len(dwarfcat.lsst_g_Err)) 
    dwarftable['flag'] = 'dw'
    dwarftable['age'] = dwarfcat.age

    table_final = vstack([dwarftable,bgtable])

    # if center_cut and save:
    #     r = float(input("Radius of cut (kpc)? "))
    #     print('Masking center ... ')
    #     posvec = np.array([table_final['c1'],table_final['c2']]).T
    #     center_mask  = np.where(
    #         (np.linalg.norm(posvec,axis=1) > r)
    #         )[0]
    #     table_cmask = table_final[center_mask]
    #     print("Saving ... ")
    #     table_cmask.write(f'../../matched_filter_testing/catalogs/{name}_formf_bg{int(edgelength)}kpc_cmask.fits',format='fits',overwrite=True)
    
    # if plot:
    #     plt.figure(figsize=(6,6))
    #     plt.scatter(dwarftable['c1'], dwarftable['c2'],s=.1,alpha=0.5)
    #     # if center_cut:
    #     #     plt.scatter(table_cmask['c1'],table_cmask['c2'],s=0.1,alpha=0.03)
    #     # else:
    #     plt.scatter(table_final['c1'],table_final['c2'],s=0.1,alpha=0.03)
    #     plt.show()

    if save : # and not center_cut:
        print("Saving ... ")
        # ascii.write(table_final,f'../../matched_filter_testing/catalogs/{name}_formf_bg{int(edgelength)}deg.txt',overwrite=True)
        table_final.write(f'../../data/matched-filter/catalogs/{name}_{c1}{c2}_formf_bg{int(edgelength)}deg.fits',format='fits',overwrite=True)

    return table_final # delete this if doing in runner.py

# kde of RGB sources, observatioal style 
# fix vmin max issue, colorbar, it looks like shit
def plot_source_map(name,mlim,D,c1,c2,path,pdist,plotdir,pix,kernel=0.7): # kernel is fixed, same as MF code 
    cat = Table.read(path, format='fits')

    if mlim == '25':
        low = 25.
    elif mlim == '26p5':
        low = 26.

    dmod = 5*np.log10(D*1e3) - 5
    TRGB = -3 # r band, from Burcin's 2021 dwarf detectability paper 

    rgbpath = Path([
        [0.6,TRGB+dmod],
        [1.1,TRGB+dmod],
        [0.9,low],
        [0.4,low],
        [0.6,TRGB+dmod]
    ]) 

    inrgb = rgbpath.contains_points(np.array([cat['gmag'] - cat['rmag'],
                                          cat['rmag']]).T)

    # selecting only rgb stars for kde
    Xpos = 60.*cat['c1'][inrgb] # deg to arcmin, already centered so the cosine and all that is nothing 
    Ypos = 60.*cat['c2'][inrgb]
    edges = np.arange(-pdist,pdist,pix)

    H, xedges, yedges = np.histogram2d(Xpos,Ypos,bins=(edges,edges)) # bins might be worng ehre but idk
    density = H.T
    gkernel = Gaussian2DKernel(x_stddev=kernel)
    z = convolve(density, gkernel)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.minorticks_on()

    cbar = ax.imshow(
        z,
        origin='lower',
        cmap='Spectral_r',
        extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]],
        
        # vmin=1,
        # vmax=5 # fiddle with this
    ) # mess with colorbar later
    # ax.set_xlim(-pdist,pdist)
    # ax.set_ylim(-pdist,pdist)
    

    savepath = f"../../data/plots/{plotdir}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    plt.savefig(f"{savepath}/{name}_{c1}{c2}_sourceKDE_rgb_{int(np.round(pdist,0))}.png", bbox_inches = 'tight')
    plt.show()
    # plt.close()
    return

# same exact process of binning+smoothing as MF code, but just counts (no weights) and stars (no bg sources)
def plot_dens_map(plotdir, name, catalog,
                    RAkey = 'c1', DECkey = 'c2',
                    numpix=2500.,pix=90.,pdist=60.):
    
    # just ananke sources
    tab = Table.read(catalog, format='fits')
    mask = np.where(tab['flag'] == 'dw')
    RA = tab[RAkey][mask]
    DEC = tab[DECkey][mask]

    RA_0 = 0 #<< in degrees
    DEC_0= 0 #<< in degrees

    yrange = [-pdist,pdist] # in arcmin 
    xrange = [-pdist,pdist]

    Xpos = 60.*(RA-RA_0)*np.cos(DEC_0/180.0*np.pi) # deg to arcmin
    Ypos = 60.*DEC-DEC_0

    xvec_pos = get_xposvector(xrange[0],xrange[1],pix)
    yvec_pos = get_yposvector(yrange[0],yrange[1],pix)

    xels = len(xvec_pos)
    yels = len(yvec_pos)

    map = np.zeros((xels,yels))
    for i in range(len(xvec_pos)):
        for j in range(len(yvec_pos)):
            xcurrmax = xvec_pos[i]+0.5*pix
            xcurrmin = xvec_pos[i]-0.5*pix
            ycurrmax = yvec_pos[j]+0.5*pix
            ycurrmin = yvec_pos[j]-0.5*pix 

            goodindex = ((Xpos < xcurrmax) & (Xpos > xcurrmin) & \
                        (Ypos < ycurrmax) & (Ypos > ycurrmin))
                
            goodcount = len(Xpos[goodindex])
    
            map[i,j] = goodcount

    kernel = Gaussian2DKernel(x_stddev=0.7) 
    smoothmap = convolve(map, kernel)
    mean, median, sigma = stats.sigma_clipped_stats(smoothmap, sigma=2, maxiters=4)

    levels = [4,5,7,10,20,50,100,150]
    
    sig_array = (smoothmap-median)/sigma
    sig_array = sig_array.T
    
    if not os.path.exists(plotdir):
            os.makedirs(plotdir)

    mapplot=plotdir+'dens_'+str(name)+f'_{np.round(pdist,0)}{RAkey}{DECkey}.png'

    plt.figure()
    plt.imshow(sig_array[::-1, :], extent=[xrange[0], xrange[1],  yrange[0], yrange[1]], \
            cmap = 'gray_r', vmin = -15, vmax = 30)
    plt.contour(xvec_pos, yvec_pos, sig_array, levels=levels)
    # plt.gca().invert_xaxis()
    plt.xlabel(r'$\Delta$x (arcmin)')
    plt.ylabel(r'$\Delta$y (arcmin)')
    # plt.xlim(-pdist,pdist)
    # plt.ylim(-pdist,pdist)
    plt.savefig(mapplot, bbox_inches = 'tight')
    plt.close()

    return