"""ASCII banner for OmniMedSeg."""

BANNER = r"""

 ██████╗ ███╗   ███╗███╗   ██╗██╗███╗   ███╗███████╗██████╗ ███████╗███████╗ ██████╗ 
██╔═══██╗████╗ ████║████╗  ██║██║████╗ ████║██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝ 
██║   ██║██╔████╔██║██╔██╗ ██║██║██╔████╔██║█████╗  ██║  ██║███████╗█████╗  ██║  ███╗
██║   ██║██║╚██╔╝██║██║╚██╗██║██║██║╚██╔╝██║██╔══╝  ██║  ██║╚════██║██╔══╝  ██║   ██║
╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║██║ ╚═╝ ██║███████╗██████╔╝███████║███████╗╚██████╔╝
 ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚══════╝╚═════╝ ╚══════╝╚══════╝ ╚═════╝ 
                                                                                     
"""

def print_banner(file=None):
    """Print the OmniMedSeg ASCII banner (to stderr by default, keeping stdout for command output)."""
    import sys

    file = file or sys.stderr
    print(BANNER, file=file)
    print("  Medical Imaging Dataset Download & Conversion Tool\n", file=file)
