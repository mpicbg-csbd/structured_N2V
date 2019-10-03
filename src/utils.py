import inspect
import difflib
import collections
from collections import Sequence

try:
  from colorama import Fore, Back, Style, init
  init()
except ImportError:  # fallback so that the imported classes always exist
  class ColorFallback():
    __getattr__ = lambda self, name: ''  
  Fore = Back = Style = ColorFallback()

def diff_func_source(f1,f2):
  """
  works with functions, classes or entire modules
  prints colored diff output to terminal
  makes it easier to put large, similar functions into same file/module
  """
  def color_diff(diff):
    for line in diff:
      if line.startswith('+'):
        yield Fore.GREEN + line + Fore.RESET
      elif line.startswith('-'):
        yield Fore.RED + line + Fore.RESET
      elif line.startswith('^'):
        yield Fore.BLUE + line + Fore.RESET
      else:
        yield line

  lines1 = inspect.getsourcelines(f1)
  lines2 = inspect.getsourcelines(f2)
  diff = color_diff([line for line in difflib.ndiff(lines1[0],lines2[0])])
  for l in diff: print(l,end='')

def flatten(l):
  for el in l:
    if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
      yield from flatten(el)
    else:
      yield el

def recursive_map(func, seq):
  def loop(func,seq):
    if isinstance(seq, (list,set,tuple)):
      for item in seq:
        yield type(item)(loop(func,item))
    elif isinstance(seq, dict):
      for k,v in seq.items():
        yield type(v)(loop(func,v))
    else:
      yield func(item)
  return type(seq)(loop(func,seq))

from collections import Collection, Mapping

def recursive_map2(func, data):
    apply = lambda x: recursive_map2(func, x)
    if isinstance(data, Mapping):
        return type(data)({k: apply(v) for k, v in data.items()})
    elif isinstance(data, Collection) and not isinstance(data, str):
        return type(data)(apply(v) for v in data)
    else:
        return func(data)

def _print_fres(names, vals, uncs, sigds = 2, rfmt = 'pm', ws = False):
  # sigds are the significance digits
  # inputs are lists of names, values and uncertainties respectively
    try:
        if all([str(u).lower() not in 'inf' for u in uncs]):
                sigs = [
                    (re.search('[1-9]', str(u)).start()-2 \
                        if re.match('0\.', str(u)) \
                    else -re.search('\.', str(float(u))).start())+sigds \
                    for u in uncs
                    ]
                # significant digits rule in uncertainty
        else:
            print('Warning: infinity in uncertainty values')
            sigs = [sigds] * len(uncs)
    except TypeError: #NaN or None
        raise TypeError('Error: odd uncertainty values')

    rfmt = rfmt.lower()
    # this can be done better/prettier I think
    if rfmt in ['fancy', 'pms']: # pms stands for pmsign
        res_str = '{{0}} = {{1:{ws}{nfmt}}} ± {{2:{ws}{nfmt}}}'
    elif rfmt in ['basic', 'pm', 'ascii']:
        res_str = '{{0}} = {{1:{ws}{nfmt}}}+/-{{2:{ws}{nfmt}}}'
    elif rfmt in ['tex', 'latex']:
        res_str = '${{0}} = {{1:{ws}{nfmt}}} \\pm {{2:{ws}{nfmt}}}$'
    elif rfmt in ['s1', 'short1']:
        res_str = '{{0}} = {{1:{ws}{nfmt}}} ± {{2:{ws}{nfmt}}}'
        # not yet supported. to do: shorthand notation
    elif rfmt in ['s2', 'short2']:
        res_str = '{{0}} = {{1:{ws}{nfmt}}}({{2:{ws}{nfmt}}})'
    else:
        raise KeyError('rfmt value is invalid')

    for i in range(len(vals)):
        try:
            print((res_str.format(
                    nfmt = '1e' if uncs[i] >= 1000 or uncs[i] <= 0.001 \
                    # 1 decimal exponent notation for big/small numbers
                        else (
                            'd' if sigs[i] <= 0 \
                            # integer if uncertainty >= 10
                            else '.{}f'.format(sigs[i])),
                    ws = ' ' if ws in [True, ' '] else ''
                    )
                 ).format(
                    names[i],
                    round(vals[i], sigs[i]),
                    round(uncs[i], sigs[i])
                    # round to allow non-decimal significances
                 )
             )

        except (TypeError, ValueError, OverflowError) as e:
            print('{} value is invalid'.format(uncs[i]))
            print(e)
            continue
    # to do: a repr method to get numbers well represented
    # instead of this whole mess