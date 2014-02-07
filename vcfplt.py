__author__ = 'alimanfoo@googlemail.com'
__version__ = '0.9-SNAPSHOT'


from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.direction'] = 'out'


def allele_balance_plot(G, AD, coverage=None, colors='bgrcmyk', legend=True, ax=None, **kwargs):
    """
    Plot allele depths coloured by genotype for a single sample. N.B., assumes biallelic variants.

    Parameters
    ---------

    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 2-dimensional array of integers with shape (#variants, 2) where
        the second axis represents depths of the first and second alleles
    coverage: int
        Maximum coverage expected (used to limit axes)
    colors: sequence
        Colors to use for hom ref, het and hom alt genotypes respectively
    legend: boolean
        If True add a legend
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # define coverage limit
    if coverage is None:
        coverage = np.percentile(AD, 98)

    # set plotting defaults
    pltargs = {
        'alpha': .05,
        'marker': 'o',
        'linestyle': ' ',
        'markeredgewidth': 0,
    }
    pltargs.update(kwargs)

    # plot each genotype separately
    states = range(np.max(G)+1)
    for g, color in zip(states, cycle(colors)):
        # include only calls with given genotype
        indices = np.nonzero(G == g)[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 0]
        Y = ADf[:, 1]
        ax.plot(X, Y, color=color, label=g, **pltargs)

    # set axis limits
    ax.set_xlim(-2, coverage)
    ax.set_ylim(-2, coverage)

    # plot diagonal
    ax.plot([0, coverage], [0, coverage], color='gray', linestyle=':')

    # make pretty
    for s in 'top', 'right', 'bottom', 'left':
        ax.spines[s].set_visible(False)
    ax.set_xlabel('ref allele depth')
    ax.set_ylabel('alt allele depth')
    ax.grid(axis='both')

    # make legend
    if legend:
        proxies = list()
        for g, color in zip(states, cycle(colors)):
            p = plt.Rectangle([0, 0], 1, 1, fc=color)
            proxies.append(p)
        ax.legend(proxies, states)

    return ax


def allele_balance_hist(G, AD, colors='bgrcmyk', bins=30, legend=True, ax=None, **kwargs):
    """
    Plot a histogram of the fraction of reads supporting the alternate allele.
    N.B., assumes biallelic variants.

    Parameters
    ---------

    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 2-dimensional array of integers with shape (#variants, 2) where
        the second axis represents depths of the first and second alleles
    colors: str or sequence
        Colors to use for hom ref, het and hom alt genotypes respectively
    bins: int
        Number of bins to use
    legend: boolean
        If True add a legend
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.hist().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .5,
        'histtype': 'bar',
        'linewidth': 0,
    }
    pltargs.update(kwargs)

    N = dict()
    states = range(np.max(G)+1)
    for g, color in zip(states, cycle(colors)):
        # include only calls with given genotype
        indices = np.nonzero((G == g) & (np.sum(AD, axis=1) > 0))[0]
        ADf = np.take(AD, indices, axis=0)
        X = ADf[:, 1] * 1. / np.sum(ADf, axis=1)
        n, _, _ = ax.hist(X, bins=np.linspace(0, 1, bins), color=color, **pltargs)
        N[g] = n

    # plot 50% line
    ax.axvline(.5, color='gray', linestyle=':')

    # make pretty
    for s in 'top', 'right', 'left':
        ax.spines[s].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.set_yticks([])
    ax.set_xlabel('alt allele fraction')
    ax.set_ylabel('frequency')

    # set axis limits based on het frequencies (genotype coded as 1)
    ax.set_ylim(0, max(N[1]) * 2)

    # make legend
    if legend:
        proxies = list()
        for g, color in zip(states, cycle(colors)):
            p = plt.Rectangle([0, 0], 1, 1, fc=color, alpha=pltargs['alpha'])
            proxies.append(p)
        ax.legend(proxies, states)

    return ax


def allele_balance_hexbin(G, AD, g=1, coverage=None, ax=None, **kwargs):
    """
    Plot allele depths for genotypes as a hexbin.

    Parameters
    ---------

    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    AD: array
        A 2-dimensional array of integers with shape (#variants, 2) where
        the second axis represents depths of the first and second alleles
    g: int
        Genotype to plot allele depths for (defaults to 1 = het)
    coverage: int
        Maximum coverage expected (used to limit axes)
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.hexbin().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    # define coverage limit
    if coverage is None:
        coverage = np.percentile(AD, 98)

    # set plotting defaults
    pltargs = {
        'extent': (0, coverage, 0, coverage),
        'gridsize': coverage/2,
    }
    pltargs.update(kwargs)

    # include only het calls
    indices = np.nonzero(G == g)[0]
    ADf = np.take(AD, indices, axis=0)
    X = ADf[:, 0]
    Y = ADf[:, 1]
    ax.hexbin(X, Y, **pltargs)

    # plot diagonal
    ax.plot([0, coverage], [0, coverage], color='gray', linestyle=':')

    # set axis limits
    ax.set_xlim(0, coverage)
    ax.set_ylim(0, coverage)

    # make pretty
    ax.set_xlabel('ref allele depth')
    ax.set_ylabel('alt allele depth')

    return ax


def variant_density_plot(POS, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of variants.

    Parameters
    ---------

    POS: array
        1-dimensional array of genome positions of variants
    window_size: int
        Window size to calculate density within
    lim: pair of ints
        Genome region to plot
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .5,
        'marker': '.',
        'color': 'm',
        'linestyle': ' ',
    }
    pltargs.update(kwargs)

    # make a histogram of positions
    bins = np.arange(0, np.max(POS), window_size)
    pos_hist, _ = np.histogram(POS, bins=bins)

    # define X and Y variables
    X = (bins[:-1] + window_size/2)
    Y = pos_hist*1./window_size

    # plot
    ax.plot(X, Y, **pltargs)

    # make pretty
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    ax.xaxis.tick_bottom()
    ax.set_xlabel('position')
    ax.set_ylabel('density')
    if lim is not None:
        ax.set_xlim(*lim)

    return ax


def genotype_density_plot(POS, G, g=1, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of calls of given genotype.

    Parameters
    ---------

    POS: array
        1-dmensional array of genome positions of variants
    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    g: int
        Genotype to plot density of (defaults to 1 = het)
    window_size: int
        Window size to calculate density within
    lim: pair of ints
        Genome region to plot
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # take only genotype calls matching selected genotype
    indices = np.nonzero(G == g)[0]
    POSg = np.take(POS, indices, axis=0)

    return variant_density_plot(POSg, window_size=window_size, lim=lim, ax=ax, **kwargs)


def variant_density_fill(POS, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of variants as a filled area.

    Parameters
    ---------

    POS: array
        1-dimensional array of genome positions of variants
    window_size: int
        Window size to calculate density within
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.fill_between().

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 2))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .5,
        'color': 'm',
        'linestyle': '-',
    }
    pltargs.update(kwargs)

    # make a histogram of positions
    bins = np.arange(0, np.max(POS), window_size)
    pos_hist, _ = np.histogram(POS, bins=bins)

    # define X and Y variables
    X = (bins[:-1] + window_size/2)
    Y = pos_hist*1./window_size

    # plot
    ax.fill_between(X, 0, Y, **pltargs)

    # make pretty
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y')
    ax.xaxis.tick_bottom()
    ax.set_xlabel('position')
    ax.set_ylabel('density')
    if lim is not None:
        ax.set_xlim(*lim)

    return ax


def genotype_density_fill(POS, G, g=1, window_size=10000, lim=None, ax=None, **kwargs):
    """
    Plot density (per bp) of calls of given genotype as a filled area.

    Parameters
    ---------

    POS: array
        1-dmensional array of genome positions of variants
    G: array
        A 1-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt)
    g: int
        Genotype to plot density of (defaults to 1 = het)
    window_size: int
        Window size to calculate density within
    lim: pair of ints
        Genome region to plot
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.plot().

    """

    # take only genotype calls matching selected genotype
    indices = np.nonzero(G == g)[0]
    POSg = np.take(POS, indices, axis=0)

    return variant_density_fill(POSg, window_size=window_size, lim=lim, ax=ax, **kwargs)


from scipy.spatial.distance import pdist, squareform


def pairwise_distance_heatmap(X, labels=None, metric='hamming', cmap='jet', ax=None):
    """
    Plot a heatmap of pairwise distances (e.g., between samples).

    Parameters
    ---------

    X: array
        2-dimensional array of shape (#variants, #samples) to use for distance calculations
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    metric: string
        Name of metric to use for distance calculations (see scipy.spatial.distance.pdist)
    cmap: colour map
        Colour map to use
    ax: axes
        Axes on which to draw

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

    D = pdist(X.T, metric)
    ax.imshow(squareform(D), interpolation='none', cmap=cmap)
    ax.set_xticks(range(X.shape[1]))
    ax.set_yticks(range(X.shape[1]))
    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels, rotation=0)

    return ax


def genotype_abundance_by_sample_bar(G, labels=None, colors='wbgrcmyk', legend=True, ax=None, **kwargs):
    """
    Plot a bar chard of genotype abundance by sample.

    Parameters
    ---------

    G: array
        2-dimensional array of genotypes coded as integers (e.g., 0 = hom ref, 1 = het, 2 = hom alt),
        of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    colors: sequence
        Colors to use for each genotype
    legend: boolean
        If True add a legend
    ax: axes
        Axes on which to draw

    All further keyword arguments are passed to ax.bar()

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'alpha': .8,
    }
    pltargs.update(kwargs)

    X = np.arange(G.shape[1])
    width = 1.
    states = np.unique(G)
    cumy = None
    for g, color in zip(states, cycle(colors)):
        Y = np.sum(G == g, axis=0) * 100. / G.shape[0]
        if cumy is None:
            ax.bar(X, Y, width, label=g, color=color, **pltargs)
            cumy = Y
        else:
            ax.bar(X, Y, width, label=g, bottom=cumy, color=color, **pltargs)
            cumy += Y

    if legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    ax.set_xticks(X + width/2)
    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90)

    ax.set_ylim(0, 100)
    ax.set_xlim(0, G.shape[1])
    ax.set_ylabel('percent')

    return ax


def calldata_by_sample_boxplot(X, labels=None, lim=None, ax=None, **kwargs):
    """
    Make a boxplot of calldata by sample (e.g., GQ, DP).

    Parameters
    ---------

    X: array
        2-dimensional array of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    lim: pair of numers
        Lower and upper limits to plot
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to ax.boxplot.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 4))
        ax = fig.add_subplot(111)

    # set plotting defaults
    pltargs = {
        'sym': '',
    }
    pltargs.update(kwargs)

    ax.boxplot(X, **pltargs)

    if lim is not None:
        ax.set_ylim(*lim)

    if labels is not None:
        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_xticklabels(labels, rotation=90)

    return ax


from matplotlib.colors import ListedColormap


def discrete_calldata_colormesh(X, labels=None, colors='wbgrcmyk', states=None, ax=None, **kwargs):
    """
    Make a meshgrid from discrete calldata (e.g., genotypes).

    Parameters
    ----------

    X: array
        2-dimensional array of integers of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    colors: sequence
        Colors to use for different values of the array
    states: sequence
        Manually specify discrete calldata states (if not given will be determined from the data)
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to ax.pcolormesh.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # determine discrete states
    if states is None:
        states = np.unique(X)
    colors = colors[:max(states)-min(states)+1]  # only need as many colors as states

    # plotting defaults
    pltargs = {
        'cmap': ListedColormap(colors),
        'norm': plt.normalize(min(states), max(states)+1),
    }
    pltargs.update(kwargs)

    ax.pcolormesh(X.T, **pltargs)
    ax.set_xlim(0, X.shape[0])
    ax.set_ylim(0, X.shape[1])

    ax.set_yticks(np.arange(X.shape[1]) + .5)
    if labels is not None:
#        labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_yticklabels(labels, rotation=0)

    return ax


def continuous_calldata_colormesh(X, labels=None, ax=None, **kwargs):
    """
    Make a meshgrid from continuous calldata (e.g., DP).

    Parameters
    ----------

    X: array
        2-dimensional array of integers or floats of shape (#variants, #samples)
    labels: sequence of strings
        Axis labels (e.g., sample IDs)
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to ax.pcolormesh.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)

    # plotting defaults
    pltargs = {
        'cmap': 'jet',
    }
    pltargs.update(kwargs)

    ax.pcolormesh(X.T, **pltargs)
    ax.set_xlim(0, X.shape[0])
    ax.set_ylim(0, X.shape[1])

    ax.set_yticks(np.arange(X.shape[1]) + .5)
    if labels is not None:
 #       labels = ['%s [%s] ' % (s, i) for (i, s) in enumerate(labels)]
        ax.set_yticklabels(labels, rotation=0)

    return ax


def genome_locator(POS, step=100, lim=None, ax=None, **kwargs):
    """
    Map variant index to genome position.

    Parameters
    ---------

    POS: array
        1-dmensional array of genome positions of variants
    step: int
        How often to draw a line
    lim: pair of ints
        Lower and upper bounds on genome position
    ax: axes
        Axes on which to draw

    Remaining keyword arguments are passed to Line2D.

    """

    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(7, 1))
        ax = fig.add_subplot(111)

    if lim is None:
        lim = 0, np.max(POS)
    start, stop = lim
    ax.set_xlim(start, stop)

    for i, pos in enumerate(POS[::step]):
        xfrom = pos
        xto = start + ((i * step * 1. / POS.size) * (stop-start))
        l = plt.Line2D([xfrom, xto], [0, 1], **kwargs)
        ax.add_line(l)

    ax.set_xlabel('position')
    ax.set_yticks([])
    ax.xaxis.tick_bottom()
    for l in 'left', 'right':
        ax.spines[l].set_visible(False)

    return ax


def bifurcation_analysis(G, focal_markers=None, core_haplotype = 0, n_markers = None, min_hap_count = 10,
                         max_marker_missingness=0.0, side='right'):
    """
    Convenience function called by haplotype_bifurcation()
    
    Determine the tree structure of a set of haplotypes around a focal core haplotype, i.e. do the analysis required
    to create input needed for haplotype_bifurcation plot.
    
    This function is loosely based on the function bifurcation.diagram from R package rehh (Mathieu Gautier and Renaud Vitalis)
    http://cran.r-project.org/web/packages/rehh/index.html
    

    Parameters
    ---------

    G: array
        2-dimensional array of haploid genotpyes coded as integers (-1 = missing, 0 = ancestral (or ref), 1 = derived (or alt)),
        of shape (#variants, #samples).
        
        Note that any sample that has any missing genotypes will be removed
    
    focal_markers: int/list/array
        The marker(s) which is (are) to be used as the focus. This can be an integer, list of integers or numpy array of integers.
        Each integer must be >= 0 and < G.shape[0]
    
    core_haplotype: int/list/array
        The genotype(s) of interest at the focal marker(s). These are as specified in the G array.
        
    n_markers: int
        The number of markers to analyse.
    
    min_hap_count: int
        The minimum number of haplotypes to attempt to plot. Mainly used for sanity checking.
        
    max_marker_missingness: float
        The maximum amount of missingness to allow for a marker. The default of 0.0 will only use markers that have no missing
        data (i.e. genotype == -1) for any samples. Setting to 1.0 will attempt to use all markers (though beware that samples
        with any missing data are subsequently removed). Should be a float between 0.0 and 1.0.
        
    side: string
        Which side of the focal marker to analyse. Valid values are 'left' or 'right'.
    

    Return
    ---------

    Returns a dict with two elements:
    
    node_haplotypes: structured array
    
        Holds all the possible haplotypes in the tree. Each record is a distinct haplotype at a distinct marker (i.e. each record is
        a branch in the tree).
        
        node_haplotypes is initialised with the element for the core haplotype, and the loops of the function add new elements,
        starting with the branches from the core haplotype, and ending at the haplotypes of the leaves of the tree (i.e. the
        markers furthest from the focal markers)
        
        Each record has 3 items:
            haplotype_string: str
                Haplotype as a string which is the concatenation of all the genotypes from the core haplotype to the given node. 
            marker_relative_index: int
                Position of this marker with respect to the (nearest) focal marker, where 0 is the (nearest) focal marker, 1 is
                the nearest marker to this, 2 is the second nearest marker and so on.
            n_haplotypes: int
                Number of samples with this haplotype at this marker
    
    branching_structure: structured array
    
        Haplotype tree structure to one side of the focal marker. Each element is a distinct node in the tree.
        
        branching_structure is initialised with no elements, and the loops of the function add new elements, starting with the
        core haplotype, and ending at the leaves of the tree (i.e. the marker furthest from the core haplotype).
        
        Each records has 2 items:
            n_branches: int
                Number of branches leading away from this node. Must be between 0 or 4, where 0 means this is a leaf node (this
                can only happen at furthest marker from core haplotype), 1 means there is no split, 2 means there is a biallelic
                split, 3 means there is a triallelic split, etc.
            node_indices: array of 4 ints
                Indices of the nodes connected to the current node. Each index refers to a different node in branching stucture.
                First n_branches elements should be non-zero, and remaining should be 0.
    
    """

    # sanity check inputs
    assert isinstance(G, np.ndarray), "G is not a numpy array"
    
    assert len(G.shape) == 2, "G is not a 2-dimensional array"
    
    assert side in ['left', 'right'], "side must be 'left' or 'right'"
    
    # Remove markers with high missingess
    low_missingness_markers = np.apply_along_axis(lambda x: np.sum(x==-1), 1, G) <= max_marker_missingness
    G = G[low_missingness_markers, :]
    
    # if no focal marker supplied, take the middle marker
    if focal_markers is None:
        focal_markers = G.shape[0] / 2
        
    if isinstance(focal_markers, int) or isinstance(focal_markers, np.integer):
        focal_markers = np.array([focal_markers])
    if isinstance(focal_markers, list):
        focal_markers = np.array(focal_markers)
    assert isinstance(focal_markers, np.ndarray), "focal_markers must be an int, list, or numpy array"

    low_missingness_indices = np.vectorize(lambda x:
        np.sum(low_missingness_markers[np.arange(x+1)]) - 1 if
            low_missingness_markers[x]==True else -1)
    focal_markers_new_indices = low_missingness_indices(focal_markers)
    low_missingness_focal_markers = np.where(focal_markers_new_indices == -1)[0]
    assert len(low_missingness_focal_markers) == 0, \
        "focal_markers %s have low missingness, perhaps rerun with higher max_marker_missingness?" % ( \
            focal_markers[low_missingness_focal_markers])
    focal_markers = focal_markers_new_indices
    
    if isinstance(core_haplotype, int) or isinstance(core_haplotype, np.integer):
        core_haplotype = np.array([core_haplotype])
    if isinstance(core_haplotype, list):
        core_haplotype = np.array(core_haplotype)
    assert isinstance(core_haplotype, np.ndarray), "core_haplotype must be an int, list, or numpy array"
    assert len(core_haplotype) in [1, len(focal_markers)], "core_haplotype has different length than focal_markers"
    
    if n_markers is None and side=='right':
        n_markers = G.shape[0] - np.max(focal_markers) + 1
    if n_markers is None and side=='left':
        n_markers = np.min(focal_markers)
    assert n_markers >= 0, "n_markers must be >= 0"
    
    assert min_hap_count >= 1, "min_hap_count must be >= 1"
    
    if side == 'left':
        assert n_markers <= np.min(focal_markers), "Too many markers on the left"
    if side == 'right':
        assert n_markers <= G.shape[0] - np.max(focal_markers) + 1, "Too many markers on the right"
    
    # Define structure of output arrays
    haplotype_dtype = 'a%d' % (n_markers+1)
    node_haplotypes_dtype = [('haplotype_string', haplotype_dtype), ('marker_relative_index', np.int), ('n_haplotypes', np.int)]
    branching_structure_dtype = [('n_branches', np.int), ('node_indices', np.int, 4)]
    
    # determine haplotype bifurcations
    if n_markers > 0:
        # create array of genotypes in markers. Note that we transpose the typical representation of a vcf file where markers
        # are rows and samples are columns, to make it more intuitive as haplotype bifurcation plots have haplotypes as rows
        haplotypes_in_core = G.transpose()[np.apply_along_axis(np.all, 1, G.transpose()[:, focal_markers] == core_haplotype), :]
        if side == 'right':
            haplo = haplotypes_in_core[:, (np.max(focal_markers) + 1):(np.max(focal_markers) + n_markers + 1)]
        else:
            if n_markers == np.min(focal_markers): # (focal_markers - 1):(focal_markers - n_markers - 1):-1 doesn't give the required values in this case
                haplo = haplotypes_in_core[:, (np.min(focal_markers) - 1)::-1]
            else:
                haplo = haplotypes_in_core[:, (np.min(focal_markers) - 1):(np.min(focal_markers) - n_markers - 1):-1]
            
        # remove any samples with missing genotypes
        haplo = haplo[np.sum(haplo==-1, axis=1) == 0, ]
        
        # sanity check we still have enough samples
        assert haplo.shape[0] >= min_hap_count, "Number of available haplotypes on the %s lower than min_hap_count" % side
                
        # Set up initial arrays
        
        # branching_structure holds the branching structure. Each element is a node in the tree. n_branches gives the number of branches
        # leading away from this node. At present this can only be 1 (if there is no split at the branch) or 2 (if there is a
        # split). node_1_index holds the index of the first node connected to the current node. node_2_index holds the index of the
        # second node connected to the current node (if there is a branch at the current node) or else 0 if there is no
        # branch. branching_structure is initialised with no elements, and the loops below add new elements.
        branching_structure = np.array([], dtype=branching_structure_dtype)
        
        # node_haplotypes holds all the possible haplotypes up to a given node in the tree. haplotype_string holds the haplotypes
        # as a string which is a concatenation of all the genotypes from the core haplotype to the given node. marker_relative_index holds the index of
        # the marker where 0 is the focal haplotype marker, 1 is the nearest marker to this, 2 is the second nearest marker and so on.
        # n holds gives the number of haplotypes at the given node. node_haplotypes is initialised with the element for the focal
        # haplotype, and the loops below add new elements.
        node_haplotypes = np.array([("", 0, haplo.shape[0]), ], dtype=node_haplotypes_dtype)
        
        # step through markers starting at the core haplotype (marker_relative_index == 0) and moving outwards
        for marker_relative_index in np.arange(n_markers) + 1:
            # how many nodes do we already have at this point? This is subsequently used to determine indexes of new nodes
            n_current_nodes = len(node_haplotypes)
            current_haplotypes = np.apply_along_axis(lambda x: ''.join(np.char.mod('%d', x)),1,haplo[:, 0:marker_relative_index])
    
            nodes_at_previous_marker = node_haplotypes[node_haplotypes['marker_relative_index'] == marker_relative_index - 1]
            for node_haplotype in nodes_at_previous_marker:
                possible_haps_vec = np.vectorize(lambda x: node_haplotype['haplotype_string'] + '%d' % x)
                possible_current_haplotypes = possible_haps_vec(np.arange(4))
                num_haps_vec = np.vectorize(lambda x: np.sum(current_haplotypes == possible_current_haplotypes[x]))
                n_each_possible_current_haplotype = num_haps_vec(np.arange(4))
                n_distinct_current_haplotypes = np.sum(n_each_possible_current_haplotype > 0)
                possible_current_haplotype_indices = np.zeros(4, dtype=np.int)
                index_amongst_possible_haplotypes = 0
                for possible_haplotype_index in range(4):
                  if n_each_possible_current_haplotype[possible_haplotype_index] > 0:
                    node_haplotypes = np.append(
                        node_haplotypes,
                        np.array(
                            [(
                              possible_current_haplotypes[possible_haplotype_index],
                              marker_relative_index,
                              n_each_possible_current_haplotype[possible_haplotype_index]), ],
                            dtype=node_haplotypes_dtype
                        )
                    )
                    possible_current_haplotype_indices[index_amongst_possible_haplotypes] = n_current_nodes
                    n_current_nodes = n_current_nodes + 1
                    index_amongst_possible_haplotypes = index_amongst_possible_haplotypes + 1
                branching_structure = np.append(
                    branching_structure,
                    np.array(
                        [(n_distinct_current_haplotypes, possible_current_haplotype_indices)],
                        dtype=branching_structure_dtype
                    )
                )
                
    # or if there is nothing to do because there are no markers...
    else:
        node_haplotypes = None
        branching_structure = None
    
    return(
        dict(
             node_haplotypes=node_haplotypes,
             branching_structure=branching_structure
        )
    )


def bifurcation_coords(node_haplotypes, branching_structure, focal_markers, POS, n_markers = None, side="right"):
    """
    Convenience function called by haplotype_bifurcation()
    
    Determine the coordinates of the haplotype tree to be plotted, i.e. do the work required to create the input for
    haplotype_bifurcation_branch().
    
    This function is loosely based on the function bifurcation.diagram from R package rehh (Mathieu Gautier and Renaud Vitalis)
    http://cran.r-project.org/web/packages/rehh/index.html
    

    Parameters
    ---------

    node_haplotypes: structured array
    
        Holds all the possible haplotypes in the tree. Each record is a distinct haplotype at a distinct marker (i.e. each record is
        a branch in the tree).
        
        node_haplotypes is initialised with the element for the core haplotype, and the loops of the function add new elements,
        starting with the branches from the core haplotype, and ending at the haplotypes of the leaves of the tree (i.e. the
        markers furthest from the focal markers)
        
        Each record has 3 items:
            haplotype_string: str
                Haplotype as a string which is the concatenation of all the genotypes from the core haplotype to the given node. 
            marker_relative_index: int
                Position of this marker with respect to the (nearest) focal marker, where 0 is the (nearest) focal marker, 1 is
                the nearest marker to this, 2 is the second nearest marker and so on.
            n_haplotypes: int
                Number of samples with this haplotype at this marker
    
    branching_structure: structured array
    
        Haplotype tree structure to one side of the focal marker. Each element is a distinct node in the tree.
        
        branching_structure is initialised with no elements, and the loops of the function add new elements, starting with the
        core haplotype, and ending at the leaves of the tree (i.e. the marker furthest from the core haplotype).
        
        Each records has 2 items:
            n_branches: int
                Number of branches leading away from this node. Must be between 0 or 4, where 0 means this is a leaf node (this
                can only happen at furthest marker from core haplotype), 1 means there is no split, 2 means there is a biallelic
                split, 3 means there is a triallelic split, etc.
            node_indices: array of 4 ints
                Indices of the nodes connected to the current node. Each index refers to a different node in branching stucture.
                First n_branches elements should be non-zero, and remaining should be 0.
    
    focal_markers: int/list/array
        The marker(s) which is (are) to be used as the focus. This can be an integer, list of integers or numpy array of integers.
        Each integer must be >= 0 and < G.shape[0]
    
    POS: array
        1-d array of genomic coordinates of the markers
    
    n_markers: int
        The number of markers to analyse.
        
    side: string
        Which side of the focal marker to analyse. Valid values are 'left' or 'right'.
    

    Return
    ---------

    Returns a 2-d array containing the X (column 0) and Y(column 1) coordinates of each node in the tree.
    
    """
    
    coords = np.zeros((node_haplotypes.shape[0], 2))
    for marker_relative_index in range(n_markers, -1, -1):
        node_indices = np.where(node_haplotypes['marker_relative_index'] == marker_relative_index)[0] # This identifies which rows are required
        if side == 'right':
            coords[node_indices, 0] = POS[np.max(focal_markers) + marker_relative_index]
        else:
            coords[node_indices, 0] = POS[np.min(focal_markers) - marker_relative_index]
        if marker_relative_index == n_markers:
            coords[node_indices, 1] = (np.arange(len(node_indices))+1)/2.0
        else:
            for node_index in node_indices: #TODO - clean up the following to allow arbitrary number of alleles
                if branching_structure[node_index]['n_branches'] == 1:
                    coords[node_index, 1] = coords[branching_structure[node_index]['node_indices'][0], 1]
                if branching_structure[node_index]['n_branches'] == 2:
                    coords[node_index, 1] = np.mean(
                                           [coords[branching_structure[node_index]['node_indices'][0], 1],
                                            coords[branching_structure[node_index]['node_indices'][1], 1]]
                                           )
                if branching_structure[node_index]['n_branches'] == 3:
                    coords[node_index, 1] = np.mean(
                                           [coords[branching_structure[node_index]['node_indices'][0], 1],
                                            coords[branching_structure[node_index]['node_indices'][1], 1],
                                            coords[branching_structure[node_index]['node_indices'][2], 1]]
                                           )
                if branching_structure[node_index]['n_branches'] == 4:
                    coords[node_index, 1] = np.mean(
                                           [coords[branching_structure[node_index]['node_indices'][0], 1],
                                            coords[branching_structure[node_index]['node_indices'][1], 1],
                                            coords[branching_structure[node_index]['node_indices'][2], 1],
                                            coords[branching_structure[node_index]['node_indices'][3], 1]]
                                           )
                if side == 'right':
                    coords[node_index, 0] = POS[np.max(focal_markers) + marker_relative_index]
                if side == 'left':
                    coords[node_index, 0] = POS[np.min(focal_markers) - marker_relative_index]
    return(coords)


def haplotype_bifurcation_branch(analysis_results, bifurcation_coords, n_markers, refsize, ax, **kwargs):
    """
    Convenience function called by haplotype_bifurcation()
    
    Plot a branch of the tree. Note that this is typically called twice by haplotype_bifurcation(), once for the left branch
    and once for the right branch.
    
    This function is loosely based on the function bifurcation.diagram from R package rehh (Mathieu Gautier and Renaud Vitalis)
    http://cran.r-project.org/web/packages/rehh/index.html
    

    Parameters
    ---------

    analysis_results: dict
        Results of running bifurcation_analysis(). This is a dict with two elements:
            
            node_haplotypes: structured array
            
                Holds all the possible haplotypes in the tree. Each record is a distinct haplotype at a distinct marker (i.e. each record is
                a branch in the tree).
                
                node_haplotypes is initialised with the element for the core haplotype, and the loops of the function add new elements,
                starting with the branches from the core haplotype, and ending at the haplotypes of the leaves of the tree (i.e. the
                markers furthest from the focal markers)
                
                Each record has 3 items:
                    haplotype_string: str
                        Haplotype as a string which is the concatenation of all the genotypes from the core haplotype to the given node. 
                    marker_relative_index: int
                        Position of this marker with respect to the (nearest) focal marker, where 0 is the (nearest) focal marker, 1 is
                        the nearest marker to this, 2 is the second nearest marker and so on.
                    n_haplotypes: int
                        Number of samples with this haplotype at this marker
            
            branching_structure: structured array
            
                Haplotype tree structure to one side of the focal marker. Each element is a distinct node in the tree.
                
                branching_structure is initialised with no elements, and the loops of the function add new elements, starting with the
                core haplotype, and ending at the leaves of the tree (i.e. the marker furthest from the core haplotype).
                
                Each records has 2 items:
                    n_branches: int
                        Number of branches leading away from this node. Must be between 0 or 4, where 0 means this is a leaf node (this
                        can only happen at furthest marker from core haplotype), 1 means there is no split, 2 means there is a biallelic
                        split, 3 means there is a triallelic split, etc.
                    node_indices: array of 4 ints
                        Indices of the nodes connected to the current node. Each index refers to a different node in branching stucture.
                        First n_branches elements should be non-zero, and remaining should be 0.

    bifurcation_coords: array
        2-d numpy array containing the X (column 0) and Y(column 1) coordinates of each node in the tree.

    n_markers: int
        The number of markers analysed.
        
    refsize: float
        The line width to use for a single haplotype.
        
    ax: axes
        Axes on which to draw
    
    Remaining keyword arguments are passed to Line2D.
    
    """

    lwd_adjust = analysis_results['node_haplotypes']['n_haplotypes'] * refsize
        
    for marker_relative_index in range(n_markers - 1, -1, -1):
        node_indices = np.where(analysis_results['node_haplotypes']['marker_relative_index'] == marker_relative_index)[0] # This identifies which rows are required
        for node_index in node_indices:
            x0 = bifurcation_coords[node_index, 0]
            y0 = bifurcation_coords[node_index, 1]
            for connected_node_index in range(analysis_results['branching_structure'][node_index]['n_branches']):
                tmp_lwd = lwd_adjust[analysis_results['branching_structure'][node_index]['node_indices'][connected_node_index]]
                x1 = bifurcation_coords[analysis_results['branching_structure'][node_index]['node_indices'][connected_node_index], 0]
                y1 = bifurcation_coords[analysis_results['branching_structure'][node_index]['node_indices'][connected_node_index], 1]
                l = plt.Line2D([x0, x1], [y0, y1], linewidth=tmp_lwd, solid_capstyle='round', **kwargs)
                ax.add_line(l)


def haplotype_bifurcation(G, POS=None, focal_markers=None, core_haplotype=0, n_markers_l=None, n_markers_r=None,
                          min_hap_count = 10, max_marker_missingness=0.0, refsize=0.1, xlim=None, ylim=None, ax=None, **kwargs):
    """
    Haplotype bifurcation plot as introduced by Sabeti et al. (http://www.nature.com/nature/journal/v419/n6909/abs/nature01140.html)
    
    This function is typically used to show extended haplotypes around a putatively causal marker. In many cases there might be a
    core haplotype consisting of a derived allele at a given SNP (focal marker) which has extended haplotypes due to recent
    positive selection, whereas the haplotypes for the ancestral allele at this same marker will have much shorter haplotypes.
    
    This function is loosely based on the function bifurcation.diagram from R package rehh (Mathieu Gautier and Renaud Vitalis)
    http://cran.r-project.org/web/packages/rehh/index.html
    

    Parameters
    ---------

    G: array
        2-dimensional array of haploid genotpyes coded as integers (-1 = missing, 0 = ancestral (or ref), 1 = derived (or alt)),
        of shape (#variants, #samples).
        
        Note that any sample that has any missing genotypes will be removed
    
    POS: array
        1-d array of genomic coordinates of the markers
    
    focal_markers: int/list/array
        The marker(s) which is (are) to be used as the focus. This can be an integer, list of integers or numpy array of integers.
        Each integer must be >= 0 and < G.shape[0]
    
    core_haplotype: int/list/array
        The genotype(s) of interest at the focal marker(s). These are as specified in the G array.
        
    n_markers_l: int
        The number of markers to analyse to the left of the focal marker(s).
    
    n_markers_r: int
        The number of markers to analyse to the right of the focal marker(s).
    
    min_hap_count: int
        The minimum number of haplotypes to attempt to plot. Mainly used for sanity checking.
        
    max_marker_missingness: float
        The maximum amount of missingness to allow for a marker. The default of 0.0 will only use markers that have no missing
        data (i.e. genotype == -1) for any samples. Setting to 1.0 will attempt to use all markers (though beware that samples
        with any missing data are subsequently removed). Should be a float between 0.0 and 1.0.
        
    refsize: float
        The line width to use for a single haplotype.
        
    xlim: pair of ints
        Genome region to plot. If None (default) will size according to POS.
        
    ylim: pair of floats
        y axis limits. If None (default) will size appropriately.
    
    ax: axes
        Axes on which to draw
    
    Remaining keyword arguments are passed to Line2D.
    
    """
    
    # sanity check inputs
    assert isinstance(G, np.ndarray), "G is not a numpy array"
    
    assert len(G.shape) == 2, "G is not a 2-dimensional array"
    
    # Remove markers with high missingess
    low_missingness_markers = np.apply_along_axis(lambda x: np.sum(x==-1), 1, G) <= max_marker_missingness
    G = G[low_missingness_markers, :]
    
    # if no positions supplied, plot markers equidistantly
    if POS is None:
        POS = np.arange(G.shape[0])
    else:
        POS = POS[low_missingness_markers]
        
    # if no focal marker supplied, take the middle marker
    if focal_markers is None:
        focal_markers = G.shape[0] / 2
        
    if isinstance(focal_markers, int) or isinstance(focal_markers, np.integer):
        focal_markers = np.array([focal_markers])
    if isinstance(focal_markers, list):
        focal_markers = np.array(focal_markers)
    assert isinstance(focal_markers, np.ndarray), "focal_markers must be an int, list, or numpy array"

    low_missingness_indices = np.vectorize(lambda x:
        np.sum(low_missingness_markers[np.arange(x+1)]) - 1 if
            low_missingness_markers[x]==True else -1)
    focal_markers_new_indices = low_missingness_indices(focal_markers)
    low_missingness_focal_markers = np.where(focal_markers_new_indices == -1)[0]
    assert len(low_missingness_focal_markers) == 0, \
        "focal_markers %s have low missingness, perhaps rerun with higher max_marker_missingness?" % ( \
            focal_markers[low_missingness_focal_markers])
    focal_markers = focal_markers_new_indices
    
    if isinstance(core_haplotype, int) or isinstance(core_haplotype, np.integer):
        core_haplotype = np.array([core_haplotype])
    if isinstance(core_haplotype, list):
        core_haplotype = np.array(core_haplotype)
    assert isinstance(core_haplotype, np.ndarray), "core_haplotype must be an int, list, or numpy array"
    assert len(core_haplotype) in [1, len(focal_markers)], "core_haplotype has different length than focal_markers"
    
    if n_markers_l is None:
        n_markers_l = np.min(focal_markers)
    if n_markers_r is None:
        n_markers_r = G.shape[0] - np.max(focal_markers) - 1
    assert n_markers_l >= 0, "n_markers_l must be >= 0"
    assert n_markers_r >= 0, "n_markers_r must be >= 0"
    
    assert min_hap_count >= 1, "min_hap_count must be >= 1"
        
    assert n_markers_l <= np.min(focal_markers), "Too many markers on the left"
    assert n_markers_r <= G.shape[0] - np.max(focal_markers) + 1, "Too many markers on the right"
    
    # perform analyses
    analysis_results_r = bifurcation_analysis(G, focal_markers, core_haplotype, n_markers_r, min_hap_count, side='right')
    analysis_results_l = bifurcation_analysis(G, focal_markers, core_haplotype, n_markers_l, min_hap_count, side='left')
    bifurcation_coords_r = bifurcation_coords(
                                              analysis_results_r['node_haplotypes'],
                                              analysis_results_r['branching_structure'],
                                              focal_markers,
                                              POS,
                                              n_markers_r,
                                              'right'
                                              )
    bifurcation_coords_l = bifurcation_coords(
                                              analysis_results_l['node_haplotypes'],
                                              analysis_results_l['branching_structure'],
                                              focal_markers,
                                              POS,
                                              n_markers_l,
                                              'left'
                                              )
    
    # recalibrate coordinates of left branch so focal marker is at same position
    if n_markers_l > 0 and n_markers_r > 0:
        bifurcation_coords_l[:, 1] = bifurcation_coords_l[:, 1] + bifurcation_coords_r[0, 1] - bifurcation_coords_l[0, 1]
 
    # set up axes
    if ax is None:
        fig = plt.figure(figsize=(14, 4))
        ax = fig.add_subplot(111)

    if xlim is None:
        xlim = np.min(POS), np.max(POS)
    start, stop = xlim
    ax.set_xlim(start, stop)
    if ylim is None:
        min_y = min(np.min(bifurcation_coords_l[:, 1]), np.min(bifurcation_coords_r[:, 1]))
        max_y = max(np.max(bifurcation_coords_l[:, 1]), np.max(bifurcation_coords_r[:, 1]))
    else:
        min_y, max_y = ylim
    ax.set_ylim(
        min_y - (max_y - min_y) * 0.05,
        max_y + (max_y - min_y) * 0.05
    )

    # create the plot
    if n_markers_r > 0:
        haplotype_bifurcation_branch(analysis_results_r, bifurcation_coords_r, n_markers_r, refsize, ax, **kwargs)
    
    if n_markers_l > 0:
        haplotype_bifurcation_branch(analysis_results_l, bifurcation_coords_l, n_markers_l, refsize, ax, **kwargs)
    
    # return the plot
    return ax
