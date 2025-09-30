import numpy as np
import struct
import os

class ThreednDepthData:
    """Class for reading in dmap binary files"""

    HAS_DEPTH: int = 1 << 0
    HAS_NORMAL: int = 1 << 1
    HAS_CONF: int = 1 << 2
    HAS_VIEWS: int = 1 << 3

    def __init__(self):
        self.magic = None
        self.flags = None
        self.image_size = ()  # width, height
        self.depth_size = ()  # width, height
        self.depth_range = ()  # min, max
        self.image_name = None
        self.neighbors = []
        self.K = []
        self.R = []
        self.C = []
        self.hsize = 0  # header size in bytes
        self.depthMap = np.array([])
        self.normals = np.array([])
        self.conf = np.array([])
        self.views = np.array([])

    def read_header(self, fin):
        fin.seek(0)
        self.magic = fin.read(2).decode("ascii")
        self.flags = struct.unpack("B", fin.read(1))[0]
        _ = fin.read(1)  # padding skip it
        self.image_size = struct.unpack("II", fin.read(8))
        self.depth_size = struct.unpack("II", fin.read(8))
        self.depth_range = struct.unpack("ff", fin.read(8))
        fname_len = struct.unpack("h", fin.read(2))[0]
        self.image_name = fin.read(fname_len).decode("ascii")
        num_neighbors = struct.unpack("i", fin.read(4))[0]
        self.neighbors = struct.unpack(f"{num_neighbors}I", fin.read(4 * num_neighbors))
        self.K = struct.unpack("9d", fin.read(9 * 8))
        self.R = struct.unpack("9d", fin.read(9 * 8))
        self.C = struct.unpack("3d", fin.read(3 * 8))
        self.hsize = self.headersize()

    def read_dmap(self, fin):
        if self.hsize == 0:
            self.read_header(fin)
        if self.hsize == 0:
            return
        area = self.depth_size[0] * self.depth_size[1]
        if area == 0 or not (self.flags & self.HAS_DEPTH):
            return
        fin.seek(self.hsize)
        if (area > 0) and (self.flags & self.HAS_DEPTH):
            self.depthMap = np.array(
                struct.unpack(f"{area}f", fin.read(4 * area)), dtype=np.float32
            )

    def read_normals(self, fin):
        if self.hsize == 0:
            self.read_header(fin)
        if self.hsize == 0:
            return
        area = self.depth_size[0] * self.depth_size[1]
        if area == 0 or not (self.flags & self.HAS_NORMAL):
            return
        npos = self.hsize
        if self.flags & self.HAS_DEPTH:
            npos += self.depth_size[0] * self.depth_size[1] * 4  # floats
        fin.seek(npos)
        narea = area * 3
        if self.flags & self.HAS_NORMAL:
            self.normals = np.array(
                struct.unpack(f"{narea}f", fin.read(4 * narea)), dtype=np.float32
            )

    def read_conf(self, fin):
        if self.hsize == 0:
            self.read_header(fin)
        if self.hsize == 0:
            return
        area = self.depth_size[0] * self.depth_size[1]
        if area == 0 or not (self.flags & self.HAS_CONF):
            return
        cpos = self.hsize
        if self.flags & self.HAS_DEPTH:
            cpos += self.depth_size[0] * self.depth_size[1] * 4  # 1 float per depth
        if self.flags & self.HAS_NORMAL:
            cpos += (
                self.depth_size[0] * self.depth_size[1] * 3 * 4
            )  # 3 floats per normal
        fin.seek(cpos)
        self.conf = np.array(
            struct.unpack(f"{area}f", fin.read(4 * area)), dtype=np.float32
        )

    def read_views(self, fin):
        if self.hsize == 0:
            self.read_header(fin)
        if self.hsize == 0:
            return
        area = self.depth_size[0] * self.depth_size[1]
        if area == 0 or not (self.flags & self.HAS_VIEWS):
            return
        vpos = self.hsize
        if self.flags & self.HAS_DEPTH:
            vpos += self.depth_size[0] * self.depth_size[1] * 4  # 1 float per depth
        if self.flags & self.HAS_NORMAL:
            vpos += (
                self.depth_size[0] * self.depth_size[1] * 3 * 4
            )  # 3 floats per normal
        if self.flags & self.HAS_CONF:
            vpos += (
                self.depth_size[0] * self.depth_size[1] * 4
            )  # 1 float per confidence
        fin.seek(vpos)
        self.views = np.array(
            struct.unpack(f"{area * 4}B", fin.read(4 * area)), dtype=np.uint8
        )

    def load(self, inpath):
        try:
            with open(inpath, "rb") as fin:
                self.read_header(fin)
                self.read_dmap(fin)
                self.read_normals(fin)
                self.read_conf(fin)
                self.read_views(fin)
            return True
        except Exception:
            return False

    def save(self, outpath):
        try:
            with open(outpath, "wb") as fout:
                fout.seek(0)
                fout.write(self.magic.encode("ascii"))
                fout.write(struct.pack("B", self.flags))
                fout.write(b"\x00")  # padding
                fout.write(struct.pack("II", *self.image_size))
                fout.write(struct.pack("II", *self.depth_size))
                fout.write(struct.pack("ff", *self.depth_range))
                fout.write(struct.pack("h", len(self.image_name)))
                fout.write(self.image_name.encode("ascii"))
                fout.write(struct.pack("i", len(self.neighbors)))
                fout.write(struct.pack(f"{len(self.neighbors)}I", *self.neighbors))
                fout.write(struct.pack("9d", *self.K))
                fout.write(struct.pack("9d", *self.R))
                fout.write(struct.pack("3d", *self.C))
                if self.flags & self.HAS_DEPTH:
                    fout.write(
                        struct.pack(f"{len(self.depthMap)}f", *self.depthMap.tolist())
                    )
                if self.flags & self.HAS_NORMAL:
                    fout.write(
                        struct.pack(f"{len(self.normals)}f", *self.normals.tolist())
                    )
                if self.flags & self.HAS_CONF:
                    fout.write(struct.pack(f"{len(self.conf)}f", *self.conf.tolist()))
                if self.flags & self.HAS_VIEWS:
                    fout.write(struct.pack(f"{len(self.views)}I", *self.views.tolist()))
                return True
        except Exception:
            return False

    def __bool__(self):
        return not (
            self.magic != "DR"
            or not self.has_depths()
            or (min(self.depth_size) <= 0)
            or (self.image_size[0] < self.depth_size[0])
            or (self.image_size[1] < self.depth_size[1])
        )

    def has_depths(self):
        return (self.flags & self.HAS_DEPTH) > 0

    def has_normals(self):
        return (self.flags & self.HAS_NORMAL) > 0

    def has_confidences(self):
        return (self.flags & self.HAS_CONF) > 0

    def has_views(self):
        return (self.flags & self.HAS_VIEWS) > 0

    def remove_flag_depths(self):
        if self.has_depths():
            self.flags &= ~self.HAS_DEPTH

    def remove_flag_normals(self):
        if self.has_normals():
            self.flags &= ~self.HAS_NORMAL

    def remove_flag_confidences(self):
        if self.has_confidences():
            self.flags &= ~self.HAS_CONF

    def remove_flag_views(self):
        if self.has_views():
            self.flags &= ~self.HAS_VIEWS

    def remove_depths(self):
        self.depthMap = np.array([])
        self.remove_flag_depths()

    def remove_normals(self):
        self.normals = np.array([])
        self.remove_flag_normals()

    def remove_conf(self):
        self.conf = np.array([])
        self.remove_flag_confidences()

    def remove_viewss(self):
        self.views = np.array([])
        self.remove_flag_views()

    def headersize(self):
        # magic (2) + flags (1) + padding (1) + image_size (2*4) + depth_size (2*4) + depth_range (2*4)
        hsize = 2 + 1 + 1 + 2 * 4 + 2 * 4 + 2 * 4
        # len (2) + image name
        hsize += 2 + len(self.image_name)
        # len (4) + neighbor list
        hsize += 4 + 4 * len(self.neighbors)
        hsize += (9 + 9 + 3) * 8  # K(3x3) + R(3x3) + C(1x3) doubles
        return hsize

    def filesize(self):
        # magic (2) + flags (1) + padding (1) + image_size (2*4) + depth_size (2*4) + depth_range (2*4)
        fsize = self.headersize()

        if self.has_depths():
            fsize += self.depth_size[0] * self.depth_size[1] * 4  # floats

        if self.has_normals():
            fsize += self.depth_size[0] * self.depth_size[1] * 3 * 4  # 3 floats

        if self.has_confidences():
            fsize += self.depth_size[0] * self.depth_size[1] * 4  # floats

        if self.has_views():
            fsize += self.depth_size[0] * self.depth_size[1] * 4  # 4 bytes

        return fsize

    @staticmethod
    def is_valid(fname):
        valid = False
        try:
            with open(fname, "rb") as fin:
                dmap = ThreednDepthData()
                dmap.read_header(fin)
                valid = bool(dmap) and os.path.getsize(fname) == dmap.filesize()
        except Exception:
            pass
        return valid
