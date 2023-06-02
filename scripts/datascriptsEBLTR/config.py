import matplotlib.pyplot as plt 
import itertools

desiredGradFair=["TopK","ExploreK","FairCo","FairK(Ours)","MCFair(Ours)","ILP","LP",]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors_list = prop_cycle.by_key()['color']
colors=itertools.cycle(colors_list)
marker = itertools.cycle(('v', '^', "<",">","p",'x',"X","D", 'o', '*')) 

# desiredGradFairColor={"TopK":"red","ExploreK":"orange","FairK(Ours)":"green",\
#                       "MCFair(Ours)":"blue","FairCo":"purple","ILP":"brown","LP":"pink"}
desiredColor={}
desiredMarker={}
desiredColorlist=["EBRank(Ours)","UCBRank","DBGD","MGD","PDGD","CFRandomK","CFTopK","CFEpsilon","PRIOR","BM25",]
for method in desiredColorlist:
    desiredColor[method]=next(colors)
    desiredMarker[method]=next(marker)
    
colors_list = prop_cycle.by_key()['color']
colors=itertools.cycle(colors_list)
desiredColorAba={}
desiredMarkerAba={}
desiredColorlistAba=["EBRank(Ours)","W/o-Explo.","Only-Prior","Only-Behav."]
marker = itertools.cycle(('v', '^', "<",">","p",'x',"X","D", 'o', '*')) 
for method in desiredColorlistAba:
    desiredColorAba[method]=next(colors)
    desiredMarkerAba[method]=next(marker)

def reorder(desiredList,curList):
    """
    This function index mapping from the curList according to desiredList.
    """
    reoderIndex=[]
    for legend in desiredList:
        reoderIndex.append(curList.index(legend))
    return reoderIndex
def reorderLegend(desiredLegend,ax):
    """
    This function index reorder the legend to desiredLegend.
    """
    handles, labels = plt.gca().get_legend_handles_labels()
    order=reorder(desiredLegend,labels)
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib import scale
import numpy as np
forwad=lambda x: -np.log((201-x*0.95))
inverse=lambda x: (201-np.exp(-x))/0.95
tran=scale.FuncTransform(forwad,inverse)
class Mylog2f(scale.ScaleBase):
    """
    The default linear scale.
    """

    name = 'mylog2f'

    def __init__(self, axis, **kwargs):
        # This method is present only to prevent inheritance of the base class'
        # constructor docstring, which would otherwise end up interpolated into
        # the docstring of Axis.set_scale.
        """
        """
        super().__init__(axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters to reasonable defaults for
        linear scaling.
        """
        axis.set_major_locator(scale.AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(scale.NullFormatter())
        # update the minor locator for x and y axis based on rcParams
#         if (axis.axis_name == 'x' and rcParams['xtick.minor.visible']
#             or axis.axis_name == 'y' and rcParams['ytick.minor.visible']):
#             axis.set_minor_locator(AutoMinorLocator())
#         else:
#             axis.set_minor_locator(NullLocator())
        # if axis.axis_name == 'x':
        #     axis.set_major_locator(plt.MaxNLocator(4))
        #     axis.set_major_formatter(FormatStrFormatter('%.2f'))
        # elif axis.axis_name == 'y':
        #     axis.set_major_locator(plt.MaxNLocator(5))
        #     axis.set_major_formatter(FormatStrFormatter('%.1f'))
        # else:
        #     axis.set_minor_locator(scale.NullLocator())
        # axis.set_major_locator(plt.MaxNLocator(4))
    def get_transform(self):
        """
        The transform for linear scaling is just the
        :class:`~matplotlib.transforms.IdentityTransform`.
        """
        return tran


