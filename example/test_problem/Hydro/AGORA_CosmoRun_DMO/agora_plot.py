#####################################################################
#
#  UNIFIED ANALYSIS SCRIPT EXAMPLE FOR THE AGORA PROJECT
#
#  FOR SCRIPT HISTORY SEE VERSION CONTROL CHANGELOG
#
#####################################################################

# pip install numpy==1.20.3
# pip install matplotlib==3.3.4
# pip install yt==3.6.1
# pip install pillow==8.4.0
# pip install h5py

import sys, os
for spec in ["~/yt/yt-3.0", "~/yt-3.0"]:
    if os.path.isdir(os.path.expanduser(spec)):
        sys.path.insert(0, os.path.expanduser(spec))
        break

if not os.path.isdir("images"): os.makedirs("images")
import h5py
import yt
from yt.config import ytcfg; ytcfg["yt","loglevel"] = "20"
# from yt.mods import *
import yt.utilities.physical_constants as phys_const
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from yt.data_objects.particle_filters import \
        particle_filter, filter_registry
# from yt.data_objects.particle_fields import \
        # particle_deposition_functions
# from yt.frontends.stream.data_structures import \
        # load_particles
# from yt.data_objects.particle_fields import \
#     particle_deposition_functions, \
#     particle_vector_functions
from yt.startup_tasks import YTParser, unparsed_args
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog

@particle_filter("finest", ["ParticleMassMsun"])
def finest(pfilter, data):
    return data["ParticleMassMsun"] < 340000

output_functions = {}
def register_output_function(func):
    output_functions[func.__name__[3:]] = func

@register_output_function
def do_gamer():
    hc = HaloCatalog()
    ds_gamer = yt.load("../Data000010")
    center = ds_gamer.argmax(("all", "density"))
    ds_gamer.add_particle_fileter("finest")
    process_dataset(ds_gamer, center)

def process_dataset(ds, center):

    #=======================
    #  [1] TOTAL MASS
    #=======================
    sp = ds.h.sphere(center, (1.0, 'mpc'))
    total_particle_mass = sp.quantities["TotalQuantity"]( ("all","ParticleMassMsun") )[0]
    print("Total particle mass within a radius of 1 Mpc of the center: %0.3e Msun" \
          % total_particle_mass)
    
    #=======================
    #  [2] PLOTS
    #=======================
    for ptype in ["finest", "all"]:
        p = ProjectionPlot(ds, "z", ("deposit", "%s_density" % ptype), center = center)
        p.save("./images/%s_z1_%s.png" % (ds, ptype))
        p.zoom(60)
        p.save("./images/%s_z2_%s.png" % (ds, ptype))

    #=======================
    #  [3] PLOTS-2
    #=======================
    w = (1.0/0.702, "mpc")
    #w = (1.0, "mpch")
    #w = (0.2/0.702, "mpc")
    axis = 2
    colorbounds = (1e-32, 1e-25)
    #colorbounds = (1e-54, 1e-48)
    res = [1024] * 3
    res[axis] = 16

    LE = center - 0.5*(w[0]/ds[w[1]])
    RE = center + 0.5*(w[0]/ds[w[1]])
    source = ds.h.arbitrary_grid(LE, RE, res)

    field = ("deposit", "all_cic")
    num = (source[field] * source[field]).sum(axis=axis)
    #num = (source[field] * source[field] * source[field]).sum(axis=axis)
    num *= (RE[axis] - LE[axis])*ds['cm'] # dl
    den = (source[field]).sum(axis=axis)
    den *= (RE[axis] - LE[axis])*ds['cm'] # dl
    proj = (num/den)
    proj[proj!=proj] = 1e-100 # remove NaN's
    plt.clf()
    norm = LogNorm(colorbounds[0], colorbounds[1], clip=True)
    plt.imshow(proj.swapaxes(0,1), interpolation='nearest', origin='lower',
               norm = norm, extent = [-0.5*(w[0]/ds[w[1]]), 0.5*(w[0]/ds[w[1]]), 
                                       -0.5*(w[0]/ds[w[1]]), 0.5*(w[0]/ds[w[1]])])
    cb = plt.colorbar()
    cb.set_label(r"$\mathrm{Density}\/\/[\mathrm{g}/\mathrm{cm}^3]$")
    plt.savefig("./images/%s_%s.png" % (ds, field[1]), dpi=150, bbox_inches='tight', \
                pad_inches=0.1)
    
    #=======================
    #  [4] PROFILES
    #=======================
    sphere_radius        = 300  # kpc
    inner_radius         = 0.8  # kpc
    total_bins           = 30

    sp = ds.h.sphere(center, (sphere_radius, 'kpc'))
    prof = BinnedProfile1D(sp, total_bins, "ParticleRadiuskpc", \
                           inner_radius, sphere_radius, end_collect = True)
    prof.add_fields([("all","ParticleMassMsun")],
                    weight = None, accumulation=False)
    prof["AverageDMDensity"] = prof[("all","ParticleMassMsun")] * \
                               phys_const.mass_sun_cgs / (phys_const.cm_per_kpc)**3 # g/cm^3
    shell_volume_temp = prof["ParticleRadiuskpc"]**3.0 * (4.0/3.0)*np.pi
    shell_volume      = prof["ParticleRadiuskpc"]**3.0 * (4.0/3.0)*np.pi
    shell_volume[1:] -= shell_volume_temp[0:-1]
    prof["AverageDMDensity"] /= shell_volume

    plt.clf()
    plt.loglog(prof["ParticleRadiuskpc"], prof["AverageDMDensity"], '-k')
    plt.xlabel(r"$\mathrm{Radius}\/\/[\mathrm{kpc}]$")
    plt.ylabel(r"$\mathrm{Dark}\/\mathrm{Matter}\/\mathrm{Density}\/\/[\mathrm{g}/\mathrm{cm}^3]$")
    plt.ylim(1e-29, 1e-23)
    plt.savefig("./images/%s_radprof.png" % ds)

    fout = open("./images/%s_profile.dat" % ds, "w")
    fout.write("# sphere_radius:"+str(sphere_radius)+"\n")
    fout.write("# inner_radius:"+str(inner_radius)+"\n")
    fout.write("# \n")
    fout.write("# radius(kpc)   gas_DM_enclosed(Msun)   prof_DM_shell(g/cm^3)\n")
    fout.write("# \n")
    for k in range(0, total_bins):
        fout.write(str(prof["ParticleRadiuskpc"][k])+"   "+ 
                   str(prof[("all","ParticleMassMsun")][k])+"   "+ 
                   str(prof["AverageDMDensity"][k])+"\n")
    fout.close()

    #=======================
    #  [5] HOP HALOFINDER
    #=======================
    if os.path.exists("./images/%s_MergerHalos.out" % ds) == False:
        halos = HaloFinder(ds, threshold = 80)
        halos.write_out("./images/%s_MergerHalos.out" % ds)
        
        field = ("deposit", "finest_density")
        source = ds.h.region(center, center - (w[0]/ds[w[1]])/2.0, center + (w[0]/ds[w[1]])/2.0)
        proj = ds.h.proj(field, "z", weight_field = field, data_source = source)
        pw = proj.to_pw(fields = [field], center = center, width = w)
        pw.set_zlim(field, 1e-31, 1e-24)
        pw.annotate_hop_circles(halos, annotate=True, print_halo_mass=False, min_size=1000)
        pw.save("./images/%s_Projection_z_%s_subset.png" % (ds, field[1]))

if __name__ == '__main__':
    parser = YTParser(description = 'AGORA analysis')
    parser.add_argument("--run-all", action="store_true", dest = "run_all")
    for output_type in sorted(output_functions):
        parser.add_argument("--run-%s" % output_type,
                dest = "outputs",
                action = "append_const",
                const = output_type,
                )
    parser.add_argument("--nref", dest="n_ref", action="store", type=int,
                        default = 64)
    opts = parser.parse_args(unparsed_args)
    if opts.run_all:
        outputs = output_functions.keys()
    else:
        outputs = opts.outputs
    if outputs is None:
        parser.error("No outputs supplied.")
        sys.exit()
    NREF = opts.n_ref
    for output in sorted(outputs):  
        print("Examining %s" % output)
        output_functions[output]()

