import Pyro5.api

# look up by friendly name
ns   = Pyro5.api.locate_ns()
uri  = ns.lookup("QHY42.Camera")
cam  = Pyro5.api.Proxy(uri)

# take 5 × 10‑s exposures
path = cam.do_exposure(0.5,10, flatten = False)
print("saved to", path)