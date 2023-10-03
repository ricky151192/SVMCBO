import numpy as np
import subprocess

### To define a new test function please add the domain bounds inside of the definition of the function

def rosenbrock(x):
    maxx = np.array([1.5, 2.5])
    minn = np.array([-1.5, -0.5])
    x = x * (maxx-minn) + minn

    x1 = x[0]
    x2 = x[1]
    return((1-x1)**2) + 100 * (x2-x1**2)**2

def rosenbrock_fun_c_1(x):
    maxx = np.array([1.5, 2.5])
    minn = np.array([-1.5, -0.5])
    x = x * (maxx-minn) + minn

    x1 = x[0]
    x2 = x[1]
    if ( x1**2 + x2**2 ) <= 2.:
        return((1-x1)**2) + 100 * (x2-x1**2)**2
    else:
        return np.nan

def rosenbrock_fun_c_2(x):
    maxx = np.array([1.5, 2.5])
    minn = np.array([-1.5, -0.5])
    x = x * (maxx-minn) + minn

    x1 = x[0]
    x2 = x[1]
    c = ((x1 - 1)**3 - x2 <= -1.) and (x1 + x2 <= 2.)
    if c:
        return ((1-x1)**2) + 100 * (x2-x1**2)**2
    else:
        return np.nan

def branin(x):

    x1 = x[0]
    x2 = x[1]
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    term1 = x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y = (term1**2 + term2 - 44.81) / 51.95

    return y

def branin_c(x, a=1, b=5.1/(4*np.pi**2) ,c=5/np.pi):
    # ----- Parameters of the elliptic feasible region #1 -------------
    x1min = 0
    x1max = 1
    cx1 = (x1min + x1max) / 3
    cx2 = (x1min + x1max) / 4
    a = 0.45
    b = 0.27
    alpha = np.pi / 4
    # -------------------------------------------------------------------

    # ----- Params of the elliptic feas. region #2 (for disonnected case)
    c2x1 = 5 * (x1min + x1max) / 6
    c2x2 = 7 * (x1min + x1max) / 8
    a2 = 0.25
    b2 = 0.1
    alpha2 = 3 * np.pi / 4
    # -------------------------------------------------------------------

    x1 = x[0]
    x2 = x[1]
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    term1 = x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y = (term1**2 + term2 - 44.81) / 51.95

    check_1 = ((((x1-cx1)*np.cos(alpha)) + ((x2-cx2)*np.sin(alpha)))**2)/(a**2) + \
              ((((x2-cx2)*np.cos(alpha)) - ((x1-cx1)*np.sin(alpha)))**2)/(b**2)

    check_2 = ((((x1-c2x1)*np.cos(alpha2)) + ((x2-c2x2)*np.sin(alpha2)))**2)/(a2**2) + \
              ((((x2-c2x2)*np.cos(alpha2)) - ((x1-c2x1)*np.sin(alpha2)))**2)/(b2**2)

    if check_1 < 1 or check_2 < 1:
        return y
    else:
        return np.nan


def branin_c_disc(x, a=1, b=5.1/(4*np.pi**2) ,c=5/np.pi, disc_factor=30):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    # ----- Parameters of the elliptic feasible region #1 -------------
    x1min = 0
    x1max = 1
    cx1 = (x1min + x1max) / 3
    cx2 = (x1min + x1max) / 4
    a = 0.45
    b = 0.27
    alpha = np.pi / 4
    # -------------------------------------------------------------------

    # ----- Params of the elliptic feas. region #2 (for disonnected case)
    c2x1 = 5 * (x1min + x1max) / 6
    c2x2 = 7 * (x1min + x1max) / 8
    a2 = 0.25
    b2 = 0.1
    alpha2 = 3 * np.pi / 4
    # -------------------------------------------------------------------

    x1 = x[0]
    x2 = x[1]
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    term1 = x2bar - 5.1 * x1bar**2 / (4 * np.pi**2) + 5 * x1bar / np.pi - 6
    term2 = (10 - 10 / (8 * np.pi)) * np.cos(x1bar)
    y = (term1**2 + term2 - 44.81) / 51.95

    check_1 = ((((x1-cx1)*np.cos(alpha)) + ((x2-cx2)*np.sin(alpha)))**2)/(a**2) + \
              ((((x2-cx2)*np.cos(alpha)) - ((x1-cx1)*np.sin(alpha)))**2)/(b**2)

    check_2 = ((((x1-c2x1)*np.cos(alpha2)) + ((x2-c2x2)*np.sin(alpha2)))**2)/(a2**2) + \
              ((((x2-c2x2)*np.cos(alpha2)) - ((x1-c2x1)*np.sin(alpha2)))**2)/(b2**2)

    if check_1 < 1 or check_2 < 1:
        return y
    else:
        return np.nan

def michalewicz(x, m=10):
    maxx = np.array([np.pi, np.pi])
    minn = np.array([0, 0])
    x = x * (maxx-minn) + minn
    y = 0
    for i in np.arange(len(x)):
        y = y + np.sin(x[i]) * np.sin(((i+1)*x[i]**2)/(np.pi))**(2*m)
    return(-y)

def michalewicz_c(x, m=10):
    # maxx = np.array([np.pi, np.pi])
    # minn = np.array([0, 0])
    maxx = np.repeat(np.pi, len(x)) #np.array([np.pi, np.pi])
    minn = np.repeat(0.0, len(x)) #np.array([0, 0])
    x = x * (maxx-minn) + minn
    if x[0]**3 + x[1]**3 > 15:# and x[0]**3 + x[1]**3 < 30:  #25
        return(np.nan)
    y = 0
    for i in np.arange(len(x)):
        y = y + np.sin(x[i]) * np.sin(((i+1)*x[i]**2)/(np.pi))**(2*m)
    return -y

def dejong5(x):
    maxx = np.array([50., 50.])
    minn = np.array([-50., -50.])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    mat = "-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32;" \
          "-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32"
    A = np.matrix(mat)

    sumterm1 = np.arange(0,25)
    sumterm2 = np.power((x1 - A[0, 0:26]),6)
    sumterm3 = np.power((x2 - A[1, 0:26]),6)
    sum = np.sum(1 / (sumterm1 + sumterm2 + sumterm3))

    y = 1 / (0.002 + sum)
    return y

def dejong5_c(x):
    maxx = np.array([50., 50.])
    minn = np.array([-50., -50.])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    if x1**3-x2**3 > 20:
        return np.nan
    mat = "-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32;" \
          "-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32"
    A = np.matrix(mat)

    sumterm1 = np.arange(0,25)
    sumterm2 = np.power((x1 - A[0, 0:26]),6)
    sumterm3 = np.power((x2 - A[1, 0:26]),6)
    sum = np.sum(1 / (sumterm1 + sumterm2 + sumterm3))

    y = 1 / (0.002 + sum)
    return y

def mishra_bird_c(x):
    maxx = np.array([0., 0.])
    minn = np.array([-10., -6.5])
    #minn = np.array([-20., -13])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    term1 = np.sin(x2)*np.exp((1-np.cos(x1))**2)
    term2 = np.cos(x1)*np.exp((1-np.sin(x2))**2)
    term3 = (x1-x2)**2
    y = term1 + term2 + term3
    if (x1+5)**2 + (x2+5)**2 < 25:
        return y
    else:
        return np.nan

def mishra_bird_c_disc(x, disc_factor=50):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    #print(x)
    maxx = np.array([0., 0.])
    minn = np.array([-10., -6.5])
    #minn = np.array([-20., -13])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    term1 = np.sin(x2)*np.exp((1-np.cos(x1))**2)
    term2 = np.cos(x1)*np.exp((1-np.sin(x2))**2)
    term3 = (x1-x2)**2
    y = term1 + term2 + term3
    if (x1+5)**2 + (x2+5)**2 < 25:
        return y
    else:
        return np.nan

#Corrected form of constraints here: https://dl.acm.org/doi/abs/10.5555/2946645.3053442
# Robert B. Gramacy, Genetha A. Gray, S´ebastien Le Digabel, Herbert K. H. Lee, Pritam
# Ranjan, Garth Wells, and Stefan M. Wild. Modeling an augmented Lagrangian for
# blackbox constrained optimization. Technometrics, 58(1):1–11, 2016.
def test54(x):
  #maxx = np.array([1., 0.75])
  maxx = np.array([1., 1.])
  minn = np.array([0., 0.])
  x = x * (maxx - minn) + minn
  x1 = x[0]
  x2 = x[1]
  y = x1+x2
  c1 = 0.5*np.sin(2*np.pi*(x1**2-2*x2))+x1+2*x2-1.5
  c2 = -x1**2-x2**2+1.5
  if c1>=0 and c2>=0:
    return y
  else:
    return np.nan

def test54_disc(x, disc_factor = 30):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    #maxx = np.array([1., 0.75])
    maxx = np.array([1., 1.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]

    y = x1+x2
    c1 = 0.5*np.sin(2*np.pi*(x1**2-2*x2))+x1+2*x2-1.5
    c2 = -x1**2-x2**2+1.5
    if c1>=0 and c2>=0:
        return y
    else:
        return np.nan

def alpine2(x):
    maxx = np.array([10., 10.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = (np.sin(x1) * np.sqrt(x1)) * (np.sin(x2) * np.sqrt(x2))
    return -y

def alpine2_c(x):
    maxx = np.array([10., 10.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = (np.sin(x1) * np.sqrt(x1)) * (np.sin(x2) * np.sqrt(x2))
    c1 = (x1-5)**2 + (x2-5)**2 - 0.25
    if c1 < 17:
        return -y
    else:
        return np.nan

def alpine2_c_disc(x, disc_factor = 30):
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    maxx = np.array([10., 10.])
    minn = np.array([0., 0.])
    x = x * (maxx - minn) + minn
    x1 = x[0]
    x2 = x[1]
    y = (np.sin(x1) * np.sqrt(x1)) * (np.sin(x2) * np.sqrt(x2))
    c1 = (x1-5)**2 + (x2-5)**2 - 0.25
    if c1 < 17:
        return -y
    else:
        return np.nan

# To use this test function you have to install the Emmental-GKLS generator
# The generator can be used only if asked to the authors
# For the authors please reference to this work:
# Sergeyev, Y. D., Kvasov, D. E., & Mukhametzhanov, M. S. (2017, June). Emmental-type GKLS-based multiextremal smooth test problems with non-linear constraints. In International Conference on Learning and Intelligent Optimization (pp. 383-388). Springer, Cham. 
# Link to the paper: https://link.springer.com/chapter/10.1007/978-3-319-69404-7_35

def CGen_function(x, seed=1, hardness='simple'):

    maxx = np.repeat(1, len(x))
    minn = np.repeat(-1, len(x))
    x = x * (maxx-minn) + minn
    dim_test = len(x)
    dict_hardness = {'simple' : {2 : {'r' : 0.90, 'p' : 0.2 },
                                 3 : {'r' : 0.66, 'p' : 0.2 },
                                 4 : {'r' : 0.66, 'p' : 0.2 },
                                 5 : {'r' : 0.66, 'p' : 0.3 }},
                     'hard' : {2 : {'r' : 0.90, 'p' : 0.1 },
                               3 : {'r' : 0.90, 'p' : 0.2 },
                               4 : {'r' : 0.90, 'p' : 0.2 },
                               5 : {'r' : 0.66, 'p' : 0.2 }}}

    r = dict_hardness.get(hardness).get(dim_test).get('r')
    p = dict_hardness.get(hardness).get(dim_test).get('p')
    out = ""
    for x_el in x:
        out = out + str(x_el) + " "
    input_generator = f"CGen_function.exe 0 d {seed} 30 -1.00 {r} {p} 20 2 10 5 y {out}"

    p1 = subprocess.Popen(["cmd", "/C", input_generator],
                          stdout=subprocess.PIPE)
    line = p1.stdout.readlines()
    p1.kill()
    c = float(str(line[-2]).split(":")[-1].split("\\")[0])

    if c == 0:
        y = np.nan
    else:
        y = float(str(line[-1]).split(":")[-1].split("\\")[0])
    return y


def CGen_function_disc(x, seed=1, hardness='simple', disc_factor=30):

    ## Discrization of the function
    x = np.round(np.array(x) * disc_factor, 0)/disc_factor
    maxx = np.repeat(1, len(x))
    minn = np.repeat(-1, len(x))
    x = x * (maxx-minn) + minn
    dim_test = len(x)
    dict_hardness = {'simple' : {2 : {'r' : 0.90, 'p' : 0.2 },
                                 3 : {'r' : 0.66, 'p' : 0.2 },
                                 4 : {'r' : 0.66, 'p' : 0.2 },
                                 5 : {'r' : 0.66, 'p' : 0.3 }},
                     'hard' : {2 : {'r' : 0.90, 'p' : 0.1 },
                               3 : {'r' : 0.90, 'p' : 0.2 },
                               4 : {'r' : 0.90, 'p' : 0.2 },
                               5 : {'r' : 0.66, 'p' : 0.2 }}}

    r = dict_hardness.get(hardness).get(dim_test).get('r')
    p = dict_hardness.get(hardness).get(dim_test).get('p')
    out = ""
    for x_el in x:
        out = out + str(x_el) + " "
    input_generator = f"CGen_function.exe 0 d {seed} 30 -1.00 {r} {p} 20 2 10 5 y {out}"

    p1 = subprocess.Popen(["cmd", "/C", input_generator],
                          stdout=subprocess.PIPE)
    line = p1.stdout.readlines()
    p1.kill()
    c = float(str(line[-2]).split(":")[-1].split("\\")[0])

    if c == 0:
        y = np.nan
    else:
        y = float(str(line[-1]).split(":")[-1].split("\\")[0])
    return y


def gomez_levi(x):
    maxx = np.array([ 1.0,  1.0])
    minn = np.array([-1.0, -1.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    return 4*x1**2 - 2.1*x1**4 + x1**6 / 3 + x1*x2 - 4*x2**2 + 4*x2**4 + 1.031628453

def gomez_levi_c(x):
    maxx = np.array([ 1.0,  1.0])
    minn = np.array([-1.0, -1.0])
    x = x * (maxx-minn) + minn
    x1 = x[0]
    x2 = x[1]
    c = 2 * np.sin(2*np.pi*x2)**2 - np.sin(4*np.pi*x1) <= 1.5
    if c:
        return 4*x1**2 - 2.1*x1**4 + x1**6 / 3 + x1*x2 - 4*x2**2 + 4*x2**4 + 1.031628453
    else:
        return np.nan


search_domain = {
    'rosenbrock':        np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    'rosenbrock_fun_c_1':np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    'rosenbrock_fun_c_2':np.array([[-1.5, 1.5], [-0.5, 2.5]]),
    'michalewicz':       np.array([[0., np.pi], [0., np.pi]]),
    'michalewicz_c':     np.array([[0., np.pi], [0., np.pi]]),
    'mishra_bird':       np.array([[-10.,  0.], [-6.5,  0.]]),
    'mishra_bird_c':     np.array([[-10.,  0.], [-6.5,  0.]]),
    'mishra_bird_c_disc':np.array([[-10.,  0.], [-6.5,  0.]]),
    'alpine2':           np.array([[  0., 10.], [  0., 10.]]),
    'alpine2_c':         np.array([[  0., 10.], [  0., 10.]]),
    'alpine2_c_disc':    np.array([[  0., 10.], [  0., 10.]]),
    'gomez_levi':        np.array([[ -1.,  1.], [ -1.,  1.]]),
    'gomez_levi_c':      np.array([[ -1.,  1.], [ -1.,  1.]]),
                }
def get_bounds(func_name):
    return search_domain.get(func_name, [[0.,1.],[0.,1.]])

def scale_to_domain(x, func_name):
    bounds = get_bounds(func_name)
    return x * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

def scale_from_domain(x, func_name):
    bounds = get_bounds(func_name)
    return (x - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

minimum = {
    'rosenbrock':np.array([1.,1.]),
    'rosenbrock_fun_c_1':np.array([1.,1.]),
    'rosenbrock_fun_c_2':np.array([1.,1.]),
    'mishra_bird':np.array([-3.1302468,-1.5821422]),
    'mishra_bird_c':np.array([-3.1302468,-1.5821422]),
    'mishra_bird_c_disc':np.array([-3.1302468,-1.5821422]),
    'gomez_levi':np.array([0.08984201, -0.7126564]),
    'gomez_levi_c':np.array([0.08984201, -0.7126564]),
          }

def get_minimum(func_name):
    return minimum.get(func_name)
