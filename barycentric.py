from netgen.meshing import Pnt, FaceDescriptor, Element1D, Element2D, MeshPoint
from netgen.meshing import Mesh as ngmesh
from ngsolve import BitArray

__all__ = ['alfeld_split_2d_marked', 'alfeld_split_2d']


def alfeld_split_2d_marked(mesh, els_hasneg, els_if):
    ngsmesh = mesh
    mesh = mesh.ngmesh
    newmesh = ngmesh()
    newmesh.SetGeometry(mesh.GetGeometry())
    newmesh.dim = mesh.dim

    ne = int(ngsmesh.ne)
    nref = sum(els_hasneg)
    els_had_hasneg = BitArray(ne + 2 * nref)
    els_had_if = BitArray(ne + 2 * nref)
    els_had_hasneg.Clear, els_had_if.Clear()

    for p in mesh.Points():
        newmesh.Add(p)

    def average_pnt(lv):
        ll = len(lv)
        xd_coord = [0, 0]
        for d in [0, 1]:
            xd_coord[d] = sum([mesh.Points()[v].p[d] for v in lv]) / ll
        return Pnt(xd_coord[0], xd_coord[1], 0.0)

    newmesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    for edge in mesh.Elements1D():
        v1, v2 = edge.vertices
        newmesh.Add(Element1D(vertices=[v1, v2], index=edge.index))

    for i, bnd in enumerate(ngsmesh.GetBoundaries()):
        newmesh.SetBCName(i, bnd)

    j = 0
    for i, el in enumerate(mesh.Elements2D()):
        v1, v2, v3 = el.vertices
        if els_hasneg[i]:
            cellcenter = average_pnt(el.vertices)
            v123 = newmesh.Add(MeshPoint(cellcenter))
            newmesh.Add(Element2D(index=el.index, vertices=[v1, v2, v123]))
            newmesh.Add(Element2D(index=el.index, vertices=[v3, v1, v123]))
            newmesh.Add(Element2D(index=el.index, vertices=[v2, v3, v123]))
            els_had_hasneg[j:j + 3] = True
            if els_if[i]:
                els_had_if[j: j + 3] = True
            j += 3
        else:
            newmesh.Add(Element2D(index=el.index, vertices=[v1, v2, v3]))
            j += 1
    return newmesh, els_had_hasneg, els_had_if


def alfeld_split_2d(mesh):
    ngsmesh = mesh
    mesh = mesh.ngmesh
    newmesh = ngmesh()
    newmesh.SetGeometry(mesh.GetGeometry())
    newmesh.dim = mesh.dim

    for p in mesh.Points():
        newmesh.Add(p)

    def average_pnt(lv):
        ll = len(lv)
        xd_coord = [0, 0]
        for d in [0, 1]:
            xd_coord[d] = sum([mesh.Points()[v].p[d] for v in lv]) / ll
        return Pnt(xd_coord[0], xd_coord[1], 0.0)

    newmesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))

    for edge in mesh.Elements1D():
        v1, v2 = edge.vertices
        newmesh.Add(Element1D(vertices=[v1, v2], index=edge.index))

    for i, bnd in enumerate(ngsmesh.GetBoundaries()):
        newmesh.SetBCName(i, bnd)

    j = 0
    mapping = {}

    for i, el in enumerate(mesh.Elements2D()):
        v1, v2, v3 = el.vertices
        cellcenter = average_pnt(el.vertices)
        v123 = newmesh.Add(MeshPoint(cellcenter))
        newmesh.Add(Element2D(index=el.index, vertices=[v1, v2, v123]))
        newmesh.Add(Element2D(index=el.index, vertices=[v3, v1, v123]))
        newmesh.Add(Element2D(index=el.index, vertices=[v2, v3, v123]))
        mapping[i] = (j, j + 1, j + 2)
        j += 3

    return newmesh, mapping
