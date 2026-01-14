import Pyro5.api
import Pyro5.errors
from qhy42_camera import QHY42Camera


class CameraRPC:
    """Pyro‑exposed wrapper around a single QHY42Camera instance."""
    def __init__(self):
        self.cam = QHY42Camera()
    
    
    # ---------- RPC methods ----------
    @Pyro5.api.expose
    def do_exposure(self, exptime: float, n_imgs: int = 1, flatten = True) -> str:
        """Take n_imgs exposures of length exptime [s]; returns FITS path."""
        return self.cam.do_exposure(exptime, n_imgs, flatten = flatten)
    
    @Pyro5.api.expose
    def set_gain(self, val: int):
        self.cam.set_gain(val)
    
    @Pyro5.api.expose
    def set_offset(self, val):
        self.cam.set_offset(val)
        
    @Pyro5.api.expose
    def tec_start(self):
        self.cam.set_cooler_enabled(True)
        
    @Pyro5.api.expose
    def tec_stop(self):
        self.cam.set_cooler_enabled(False)
        
    @Pyro5.api.expose
    def set_tec_setpoint(self, val):
        self.cam.set_tec_temperature(val)
    
    @Pyro5.api.expose
    def get_status(self):
        status = {}
        status.update({"tec_temperature" : self.cam.get_temperature()})
        return status
    

def main(name: str = "QHY42.Camera"):
    # 1) create a local daemon
    daemon = Pyro5.api.Daemon()                       # *binds on localhost*
    uri = daemon.register(CameraRPC)                  # register object

    # 2) try to find a running Name Server and register there
    try:
        ns = Pyro5.api.locate_ns()                    # default localhost:9090
        ns.register(name, uri)
        print(f"[Pyro] registered as '{name}' → {uri}")
    except Pyro5.errors.NamingError:
        print("[Pyro] NameServer not found → falling back to raw URI")
        print("URI:", uri)
        print("Start a NameServer with: python -m Pyro5.nameserver")

    # 3) enter request loop
    print("[Pyro] waiting for requests …   Ctrl‑C to exit")
    daemon.requestLoop()


if __name__ == "__main__":
    main()
