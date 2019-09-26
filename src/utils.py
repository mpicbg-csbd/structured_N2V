import inspect
import difflib

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


