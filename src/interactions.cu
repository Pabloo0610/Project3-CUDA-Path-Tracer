#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ glm::vec3 checkerTexture(glm::vec2 uv, float scale) {
    int x = static_cast<int>(floorf(uv.x * scale));
    int y = static_cast<int>(floorf(uv.y * scale));

    if ((x + y) % 2 == 0) {
        return glm::vec3(1.0f, 1.0f, 1.0f);
    }
    else {
        return glm::vec3(0.0f, 0.0f, 0.0f);
    }
}

__device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
	glm::vec2 uv,
    glm::vec3 normal,
    cudaTextureObject_t* texObjs,
    const Materialz &m,
    thrust::default_random_engine &rng,
    const int& proceduralType)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

  // pure diffuse
	int texId = m.texOffset + m.baseColorTexId;
    glm::vec3 texColor;
    if (texId >= 0) {
        if (proceduralType == 1) {
            texColor = checkerTexture(uv);
        }
        else
        {
            float4 texSample = tex2D<float4>(texObjs[texId], uv.x, uv.y);
            texColor = glm::vec3(texSample.x, texSample.y, texSample.z);
		}
	}
    glm::vec3 mColor;
    if (m.isGltf) {
        mColor = texColor;
    }
    else {
        mColor = m.color;
	}
    if (m.type == MATERIAL_DIFFUSE) {
    glm::vec3 randDir = calculateRandomDirectionInHemisphere(normal, rng);
    pathSegment.ray.origin = intersect + 0.001f * randDir;
    pathSegment.ray.direction = randDir;
    pathSegment.color *= mColor;
    pathSegment.remainingBounces--;
    }
    else if (m.type == MATERIAL_SPECULAR) {
    glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
    glm::vec3 reflectDir = glm::reflect(incident, glm::normalize(normal));
    pathSegment.ray.origin = intersect + 0.001f * reflectDir;
    pathSegment.ray.direction = reflectDir;
    pathSegment.color *= m.color;
    pathSegment.remainingBounces--;
    }
}
