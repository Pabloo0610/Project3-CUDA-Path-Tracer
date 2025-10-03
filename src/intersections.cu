#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    glm::vec3& barycentric)
{
    const float EPSILON = 1e-6f;

    glm::vec3 e1 = v1 - v0;
    glm::vec3 e2 = v2 - v0;

    glm::vec3 pvec = glm::cross(r.direction, e2);
    float det = glm::dot(e1, pvec);

    if (fabs(det) < EPSILON) {
        return -1.0f;
    }

    float invDet = 1.0f / det;
    glm::vec3 tvec = r.origin - v0;

    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return -1.0f;

    glm::vec3 qvec = glm::cross(tvec, e1);
    float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;

    float t = glm::dot(e2, qvec) * invDet;
    if (t < EPSILON) return -1.0f;

    intersectionPoint = r.origin + t * r.direction;

    barycentric = glm::vec3(1 - u - v, u, v);

    normal = glm::normalize(
        barycentric.x * n0 +
        barycentric.y * n1 +
        barycentric.z * n2
    );

    return t;
}

__host__ __device__ float meshIntersectionTestV0(
    Geom geom,
    GltfMesh mesh,
	GltfPrimitive* primitives,
    Vertex* vertices,
	uint32_t* indices,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_best = 1e30f;
    glm::vec3 bestP, bestN;
    bool found = false;

    for (int p = 0; p < mesh.primCount; ++p) {
        const GltfPrimitive& prim = primitives[p + mesh.primIndexStart];
        if (!prim.hasIndices || prim.indexCount < 3) continue;

        uint32_t start = prim.firstIndex;
        uint32_t end = start + prim.indexCount;

        for (uint32_t k = start; k + 2 < end; k += 3) {
            uint32_t i0 = indices[k + 0];
            uint32_t i1 = indices[k + 1];
            uint32_t i2 = indices[k + 2];

            const glm::vec3& v0 = vertices[i0].pos;
            const glm::vec3& v1 = vertices[i1].pos;
            const glm::vec3& v2 = vertices[i2].pos;

            glm::vec3 n0 = vertices[i0].normal;
            glm::vec3 n1 = vertices[i1].normal;
            glm::vec3 n2 = vertices[i2].normal;

            if (len2(n0) == 0.f && len2(n1) == 0.f && len2(n2) == 0.f) {
                glm::vec3 faceN = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                n0 = n1 = n2 = faceN;
            }

            glm::vec3 isectP, N, bary;
            float t = triangleIntersectionTest(v0, v1, v2, n0, n1, n2, q, isectP, N, bary);

            if (t > 0.0f && t < t_best) {
                t_best = t;
                bestP = isectP;
                bestN = N;
                found = true;
            }
        }
    }

    if (!found) {
        return -1.0f;
    }

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(bestP, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(bestN, 0.f)));

    outside = true;
    if (glm::dot(normal, r.direction) > 0.0f) {
        normal = -normal;
        outside = false;
    }

    return glm::length(intersectionPoint - r.origin);
}

__host__ __device__ float meshIntersectionTestV1(
    Geom geom,
    GltfMesh mesh,
    GltfPrimitive* primitives,
    GltfTriangle* triangles,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside,
    glm::vec2& uv,
    int& matId,
    const bool& isCulling)
{

    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_best = 1e30f;
    glm::vec3 bestP, bestN;
    bool found = false;

    for (int p = 0; p < mesh.primCount; ++p) {
        const GltfPrimitive& prim = primitives[p + mesh.primIndexStart];
        if (prim.triangleCount == 0) continue;
        if (isCulling && prim.bb.valid) {
            float tNear, tFar;
            if (!intersectAABB(q, prim.bb.min, prim.bb.max, tNear, tFar)) {
                continue;
            }
        }
        for (uint32_t k = 0; k < prim.triangleCount; ++k) {
            const GltfTriangle& tri = triangles[prim.firstTriangle + k];

            glm::vec3 isectP, N, bary;
            float t = triangleIntersectionTest(
                tri.v0, tri.v1, tri.v2,
                tri.n0, tri.n1, tri.n2,
                q, isectP, N, bary
            );

            if (t > 0.0f && t < t_best) {
                t_best = t;
                bestP = isectP;
                bestN = N;
                found = true;
                uv = bary.x * tri.uv0 + bary.y * tri.uv1 + bary.z * tri.uv2;
				matId = prim.materialId;
            }
        }
    }

    if (!found) {
        return -1.0f;
    }

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(bestP, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(bestN, 0.f)));

    outside = true;
    if (glm::dot(normal, r.direction) > 0.0f) {
        normal = -normal;
        outside = false;
    }

    return glm::length(intersectionPoint - r.origin);
}

__host__ __device__ bool intersectAABB(
    const Ray& ray,
    const glm::vec3& min,
    const glm::vec3& max,
    float& tNear, float& tFar)
{
    tNear = -FLT_MAX;
    tFar = FLT_MAX;

    for (int i = 0; i < 3; i++) {
        if (fabs(ray.direction[i]) < 1e-6f) {

            if (ray.origin[i] < min[i] || ray.origin[i] > max[i])
                return false;
        }
        else {
            float t1 = (min[i] - ray.origin[i]) / ray.direction[i];
            float t2 = (max[i] - ray.origin[i]) / ray.direction[i];
            if (t1 > t2) thrust::swap(t1, t2);

            tNear = fmax(tNear, t1);
            tFar = fmin(tFar, t2);

            if (tNear > tFar) return false;
            if (tFar < 0) return false;
        }
    }
    return true;
}