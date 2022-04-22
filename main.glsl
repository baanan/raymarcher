const int KEY_LEFT  = 37;
const int KEY_UP    = 38;
const int KEY_RIGHT = 39;
const int KEY_DOWN  = 40;



//constants
float start = 0.01;
float EPSILON = 0.001;
const float end = 100.0;
vec2 rotation = vec2(0, 0);
vec2 rotationd = vec2(0, 0);
int MAX_MARCHING_STEPS = 200;


//         ---- Point light ----

struct PointLight {
    vec4 Ambient;
    vec4 Diffuse;
    vec4 Specular;
    vec3 Position;
    float Range;
    vec3 Attenuation;
    float Pad;
};
    
PointLight _pointLight = PointLight(
    vec4(0.3f, 0.3f, 0.3f, 1.0),
    vec4(0.5f, 0.5f, 0.5f, 1.0),
    vec4(0.6f, 0.6f, 0.6f, 1.0),
    vec3(0.0,5.0,0.0),
    end,
    vec3(0.0f, 0.1f, 0.0f),
    1.0
);

PointLight _ppointLight = PointLight(
    vec4(0.3f, 0.3f, 0.3f, 1.0),
    vec4(0.5f, 0.5f, 0.5f, 1.0),
    vec4(0.6f, 0.6f, 0.6f, 1.0),
    vec3(0.0,-2.0,0.0),
    end,
    vec3(0.0f, 0.1f, 0.0f),
    1.0
);




struct DirectionalLight {
    vec4 Ambient;
    vec4 Diffuse;
    vec4 Specular;
    vec3 Direction;
    float Pad;
}

_dirLight = DirectionalLight(
    vec4(0.2f, 0.2f, 0.2f, 1.0),
    vec4(0.5f, 0.5f, 0.5f, 1.0),
    vec4(0.5f, 0.5f, 0.5f, 1.0),
    normalize(vec3(0.2f, -1.0f, 0.0f)),
    1.0
);
    


//           ---- Materials ----

struct Material
{
    vec4 Ambient;
    vec4 Diffuse;
    vec4 Specular; // .w = SpecPower
    float Reflect;
    float ior;
};
  
Material _landMaterial = Material(
    vec4(0.48f, 0.77f, 0.46f, 1.0),
    vec4(0.48f, 0.77f, 0.46f, 1.0),
    vec4(0.2f, 0.2f, 0.2f, 1.0),
    0.0,
    0.0
);

Material _plainMaterial = Material(
    vec4(1.00f, 1.00f, 1.00f, 0.46f),
    vec4(1.00f, 1.00f, 1.00f, 0.46f),
    vec4(0.20f, 0.20f, 0.20f, 0.1f),
    0.0,
    0.0
);
Material _reflectiveMaterial = Material(
    vec4(1.00f, 1.00f, 1.00f, 0.46f),
    vec4(1.00f, 1.00f, 1.00f, 0.46f),
    vec4(0.20f, 0.20f, 0.20f, 0.1f),
    1.0,
    0.0
);
Material _glassMaterial = Material(
    vec4(1.00f, 1.00f, 1.00f, 0.46f),
    vec4(1.00f, 1.00f, 1.00f, 0.46f),
    vec4(0.20f, 0.20f, 0.20f, 0.1f),
    1.0,
    1.5
);

Material _wavesMaterial = Material(
    vec4(0.137f, 0.42f, 0.556f, 1.0),
    vec4(0.137f, 0.42f, 0.556f, 1.0),
    vec4(1.0f, 0.8f, 0.8f, 0.8f),
    1.0,
    1.3
);

Material _skyMaterial = Material(
    vec4(vec3(157.0, 169.0, 210.0)/255.0, 1.0),
    vec4(vec3(157.0, 169.0, 210.0)/255.0, 1.0),
    vec4(vec3(157.0, 169.0, 210.0)/255.0, 1.0f),
    0.0,
    0.0
);


//           ---- Computing lights ----

void ComputePointLight(Material mat, PointLight L, vec3 pos, vec3 normal, vec3 toEye,
                   out vec4 ambient, out vec4 diffuse, out vec4 spec)
{
    // Initialize outputs.
    ambient = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    diffuse = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    spec    = vec4(0.0f, 0.0f, 0.0f, 0.0f);

    // The vector from the surface to the light.
    vec3 lightVec = L.Position - pos;
        
    // The distance from surface to light.
    float d = length(lightVec);
    
    // Range test.
    if( d > L.Range )
        return;
        
    // Normalize the light vector.
    lightVec /= d; 
    
    // Ambient term.
    ambient = mat.Ambient * L.Ambient;    

    // Add diffuse and specular term, provided the surface is in 
    // the line of site of the light.

    float diffuseFactor = dot(lightVec, normal);

    // Flatten to avoid dynamic branching.
    if( diffuseFactor > 0.0f )
    {
        vec3 v         = reflect(-lightVec, normal);
        float specFactor = pow(max(dot(v, toEye), 0.0f), mat.Specular.w);
                    
        diffuse = diffuseFactor * mat.Diffuse * L.Diffuse;
        spec    = specFactor * mat.Specular * L.Specular;
    }

    // Attenuate
    float att = 1.0f / dot(L.Attenuation, vec3(1.0f, d, d*d));

    diffuse *= att;
    spec    *= att;
}



void ComputeDirectionalLight(Material mat, DirectionalLight L, 
                             vec3 normal, vec3 toEye,
                             out vec4 ambient,
                             out vec4 diffuse,
                             out vec4 spec)
{
    // Initialize outputs.
    ambient = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    diffuse = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    spec    = vec4(0.0f, 0.0f, 0.0f, 0.0f);

    // The light vector aims opposite the direction the light rays travel.
    vec3 lightVec = -L.Direction;

    // Add ambient term.
    ambient = mat.Ambient * L.Ambient;    

    // Add diffuse and specular term, provided the surface is in 
    // the line of site of the light.
    
    float diffuseFactor = dot(lightVec, normal);

    // Flatten to avoid dynamic branching.
    if( diffuseFactor > 0.0f )
    {
        vec3 v         = reflect(-lightVec, normal);
        float specFactor = pow(max(dot(v, toEye), 0.0f), mat.Specular.w);
                    
        diffuse = diffuseFactor * mat.Diffuse * L.Diffuse;
        spec    = specFactor * mat.Specular * L.Specular;
    }
}






//             ---- Shape distance functions ----

//distance function for box
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}


//distance function for sphere
float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}


//distance function for plane
float sdPlane( vec3 p, vec3 n )
{
  // n must be normalized
  return max(0.0,dot(p,n.xyz));
}



//             ---- Scene distance functions ----

//distance function for the scene
float sceneSDFfff(vec3 point) {
    //return len(point) - 1.0;
    return 
    min(
    	min(
    		min(
    	    	min(
    	    		max(
    	     	   		max(
        		        	sdBox(point, vec3(1,1,1)), 
    	    	        	-sdSphere(point, 1.2)
     		   	    	), 
       		 	    	-sdBox(point - vec3(0.5,0.5,0.5), vec3(1,1,1))
       		 		),
       		 		sdBox(point-vec3(4,0,0),vec3(1,1,1))
        		),
       		 	sdPlane(point-vec3(0,-3,0),vec3(0,1,0))
        	),
        	//sdPlane(point - vec3(3,0,0),vec3(-1,0,0))
            999.0
    	),
        sdSphere(point - vec3(0,5,0),1.0)
    );
    //return sdSphere(point, 1.0);
    //return sdBox(point, vec3(1,1,1));
}

float sceneSDF(vec3 point) {
    return
        min(
            min(
                min(
                    min(
                        sdBox(point - vec3(-3,0,0),vec3(1.0, 1.0, 0.1)),
                        sdSphere(point - vec3(0,0,0),1.0)
                        ),
                    sdSphere(point - vec3(3,0,0),1.0)
                    ),
                sdBox(point - vec3(0,0,-5),vec3(7,1.5,0.5))
                ),
            sdPlane(point - vec3(0,-5,0),vec3(0,1,0))
            );
        
    
}




float sceneSDFf(vec3 z) {
    vec3 nz = z;
    nz.xz = mod((nz.xz),1.0)-vec2(0.5); // instance on xy-plane
    return min(length(nz)-0.3,sdPlane(z-vec3(0,-3,0),vec3(0,1,0)));             // sphere DE
    
}

float sceneSDFff(vec3 z) {
    vec3 a1 = vec3(1,1,1);
	vec3 a2 = vec3(-1,-1,1);
	vec3 a3 = vec3(1,-1,-1);
	vec3 a4 = vec3(-1,1,-1);
	vec3 c;
	int n = 0;
    float Scale = 200.0;
	float dist, d;
	while (n < 5) {
		 c = a1; dist = length(z-a1);
	        d = length(z-a2); if (d < dist) { c = a2; dist=d; }
		 d = length(z-a3); if (d < dist) { c = a3; dist=d; }
		 d = length(z-a4); if (d < dist) { c = a4; dist=d; }
		z = Scale*z-c*(Scale-1.0);
		n++;
	}

	return length(z) * pow(Scale, float(-n));
}



//   ----Estimating the normals----

vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

//      ---- getting colors (old) ----

vec3 agetColor(vec3 point) {
    if (
        max(
       	    max(
       	        sdBox(point, vec3(1,1,1)), 
       	        -sdSphere(point, 1.2)
       	    ), 
       	    -sdBox(point - vec3(0.5,0.5,0.5), vec3(1,1,1))
        ) <= EPSILON
    ) {return vec3(184.21/255.0, 35.09/255.0, 40.06/255.0);}
    if (sdBox(point-vec3(4,0,0),vec3(1,1,1)) <= EPSILON) {return vec3(58.55/255.0, 85.23/255.0, 154.6/255.0);}
    if (sdPlane(point-vec3(0,-3,0),vec3(0,1,0)) <= EPSILON) {return vec3(51.23/255.0, 157.87/255.0, 61.89/255.0);}
    if (sdPlane(point - vec3(6,0,0),vec3(-1,0,0)) <= EPSILON) {return vec3(0.75, 0.75, 0.75);}
    return vec3(0,0.1,0.2);

    
}
vec3 getColorf(vec3 z) {
    vec3 nz = z;
    nz.xz = mod((nz.xz),1.0)-vec2(0.5);
    if(length(nz)-0.3<= EPSILON) {return vec3(1,0,0);}
    if(sdPlane(z-vec3(0,-3,0),vec3(0,1,0))<= EPSILON) {return vec3(0.5,0.5,0.5);}
}




//            ---- Getting materials ----

Material getMatf(vec3 z) {
    vec3 nz = z;
    nz.xz = mod((nz.xz),1.0)-vec2(0.5);
    if(length(nz)-0.3<= EPSILON) {return _wavesMaterial;}
    if(sdPlane(z-vec3(0,-3,0),vec3(0,1,0))<= EPSILON) {return _landMaterial;}
}
Material getMatff(vec3 point) {
    if(
    	min(
    		min(
    	    	max(
    	     	   	max(
        		        sdBox(point, vec3(1,1,1)), 
    	    	        -sdSphere(point, 1.2)
     		   	    ), 
       		 	    -sdBox(point - vec3(0.5,0.5,0.5), vec3(1,1,1))
       		 	),
       		 	sdBox(point-vec3(4,0,0),vec3(1,1,1))
        	),
            sdSphere(point - vec3(0,5,0),1.0)
        ) <= EPSILON) {return _plainMaterial;}
    if(sdPlane(point-vec3(0,-3,0),vec3(0,1,0)) <= EPSILON) {return _landMaterial;}
    return _skyMaterial;
}

Material getMat(vec3 point) {
    if(sdBox(point - vec3(-3,0,0),vec3(1.0, 1.0, 0.1))<=EPSILON) {return _glassMaterial;}
    if(sdSphere(point - vec3(0,0,0),1.0)<=EPSILON) {return _reflectiveMaterial;}
    if(sdSphere(point - vec3(3,0,0),1.0)<=EPSILON) {return _plainMaterial;}
    if(sdBox(point - vec3(0,0,-5),vec3(7,1.5,0.5))<=EPSILON) {return _plainMaterial;}
    if(sdPlane(point - vec3(0,5,0),vec3(0,1,0))<=EPSILON) {return _plainMaterial;}
    
    return _skyMaterial;
}

    
//          ---- the raymarch algoritm ----

//raymarch algoritm, returns depth from start and most recent distance
vec3 raymarch(vec3 eye, vec3 viewRayDirection, float endd) {
    viewRayDirection = normalize(viewRayDirection);
    vec3 returning = vec3(0.0,0.0,0.0);
    float depth = start;
    returning.x = end;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
    	float dist = sceneSDF(eye + (depth * viewRayDirection));
    	if (dist < EPSILON) {
    	    // We're inside the scene surface!
    	    returning.x = depth;
            returning.y = dist;
            break;
    	}
    	// Move along the view ray
    	depth += dist;
	
    	if (depth >= endd || depth >= end) {
    	    // Gone too far; give up
    	    returning.x = endd;
            returning.y = dist;
            break;
    	}
        //if (i+1 == MAX_MARCHING_STEPS) {returning.z = 1.0;}
	}
    return returning;
	
}







vec3 mraymarch(vec3 eye, vec3 viewRayDirection, float endd) {
    viewRayDirection = normalize(viewRayDirection);
    vec3 returning = vec3(0.0,0.0,0.0);
    float depth = start;
    returning.x = end;
    Material pm = getMat(eye+viewRayDirection*depth);
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        
    	//float dist = sceneSDF(eye + (depth * viewRayDirection));
        float dist = 0.1;
        pm = getMat(eye+viewRayDirection*depth);
        depth += dist;
        //pm = getMat(eye+viewRayDirection*depth);
        //float dist = 0.1;
    	if (getMat(eye+viewRayDirection*depth) != pm) {
    	    // We're inside the scene surface!
    	    returning.x = depth;
            returning.y = dist;
            break;
    	}
    	// Move along the view ray
	
    	if (depth >= endd || depth >= end) {
    	    // Gone too far; give up
    	    returning.x = endd;
            returning.y = dist;
            break;
    	}
        
        //if (i+1 == MAX_MARCHING_STEPS) {returning.z = 1.0;}
	}
    return returning;
	
}






void swap (out float a, out float b) {
    float c = a;
    a = b;
    b = c;
}






float FresnelReflectAmount (float n1, float n2, vec3 normal, vec3 incident)
{
        // Schlick aproximation
        float r0 = (n1-n2) / (n1+n2);
        r0 *= r0;
        float cosX = -dot(normal, incident);
        if (n1 > n2)
        {
            float n = n1/n2;
            float sinT2 = n*n*(1.0-cosX*cosX);
            // Total internal reflection
            if (sinT2 > 1.0)
                return 1.0;
            cosX = sqrt(1.0-sinT2);
        }
        float x = 1.0-cosX;
        float ret = r0+(1.0-r0)*x*x*x*x*x;
 
        // adjust reflect multiplier for object reflectivity
        //ret = (OBJECT_REFLECTIVITY + (1.0-OBJECT_REFLECTIVITY) * ret);
        return ret;
}

vec3 ref(const vec3 I, const vec3 N, const float ior) 
{ 
    float cosi = clamp(dot(I, N), -1.0, 1.0); 
    float etai = 1.0, etat = ior; 
    vec3 n = N; 
    if (cosi < 0.0) { cosi = -cosi; } else { swap(etai, etat); n= -N; } 
    float eta = etai / etat; 
    float k = 1.0 - eta * eta * (1.0 - cosi * cosi); 
    if(k < 0.0) { return vec3(0.0); } else { return eta * I + (eta * cosi - sqrt(k)) * n; }
    //return k < 0.0 ? 0.0 : eta * I + (eta * cosi - sqrt(k)) * n; 
} 













bool plInShadow(vec3 point, vec3 light) {
    vec3 hitpos = point;
    vec3 dir = light - hitpos;
    vec3 dirn = normalize(dir);
    vec3 offset = dirn * (sceneSDF(hitpos));
    
    vec3 dist2 = raymarch(hitpos + offset, dirn, distance(vec3(0),dir));
    return dist2.x<distance(vec3(0),dir);
    
}












vec3 getColor(vec3 eye,vec3 hitpos) {
    float dist = length(hitpos - eye);
    
    // Start with a sum of zero. 
    vec4 ambient = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    vec4 diffuse = vec4(0.0f, 0.0f, 0.0f, 0.0f);
    vec4 spec    = vec4(0.0f, 0.0f, 0.0f, 0.0f);

    // Sum the light contribution from each light source.
    vec4 A, D, S;
    
    vec3 toEyel = normalize(eye - hitpos);
    vec3 color = vec3(0.0);
    Material matNow = getMat(hitpos);

    ComputePointLight(matNow,_pointLight,hitpos,estimateNormal(hitpos),toEyel, A, D, S);
    ambient += A;
    if(!plInShadow(hitpos,_pointLight.Position)) {
    	diffuse += D;
    	spec    += S;
    }
    
    
    //Material mat, DirectionalLight L, vec3 normal, vec3 toEye, out vec4 ambient, out vec4 diffuse,out vec4 spec
    //ComputeDirectionalLight(matNow,_dirLight,estimateNormal(hitpos),toEyel,A,D,S);
    //ambient += A;
    //if(!plInShadow(hitpos,hitpos-(_dirLight.Direction*end))) {
    //	diffuse += D;
    //	spec    += S;
    //}
    
    float divideby = 1.0;
    ambient /= divideby;
    diffuse /= divideby;
    spec /= divideby;
    
    //ComputePointLight(matNow,_ppointLight,hitpos,estimateNormal(hitpos),toEyel, A, D, S);
    //ambient += A;
    //if(!inShadow(hitpos,__pointLight.Position)) {
    	//diffuse += D;
    	//spec    += S;
    //}
    
    vec4 litColor = ambient + diffuse + spec;

    color = ambient.xyz;
    
    
    //if(dist<end) {
        //if(!plInShadow(hitpos, _pointLight.Position)) {
            //color = vec3(0.05);
            //color = litColor.xyz;
            //color = ambient.xyz;
        //}
        //if(!inShadow(hitpos, _ppointLight.Position)) {
        //    //color = vec3(0.05);
        //    color = litColor.xyz;
            //color = ambient.xyz;
        //}
    //}
    color = litColor.xyz;
    return color;
}











//multiplies two vectors
vec3 VecMatMult(mat3 matrix, vec3 vector) {
    float x = matrix[0].x*vector.x +
        	  matrix[0].y*vector.y +
        	  matrix[0].z*vector.z;
    float y = matrix[1].x*vector.x +
        	  matrix[1].y*vector.y +
        	  matrix[1].z*vector.z;
    float z = matrix[2].x*vector.x +
        	  matrix[2].y*vector.y +
        	  matrix[2].z*vector.z;
    return vec3(x,y,z);
}

float len(float x, float y) {
    return sqrt(x*x+y*y);
}















//drawing stuff
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    
    
    //constant things
    //rotation.x = iTime*0.5;
    rotation.x = sin(iTime)/1.0;
    //rotation.y = sin(iTime*3.0)/3.0;
    rotation.y = -0.1;
    
    float M_PI = 3.14159265359;
    float width = iResolution.x, height = iResolution.y; 
    float invWidth = 1.0 / float(width), invHeight = 1.0 / float(height); 
    float fov = 30.0, aspectratio = float(width) / float(height); 
    float angle = tan(fov/2.0);
    float x = fragCoord.x;
    float y = fragCoord.y;
    //vec3 eye = vec3(sin(rotation.x)*5.0,-sin(iTime*3.0)*2.0,cos(rotation.x)*5.0);
    vec3 eye = vec3(sin(rotation.x)*5.0,2,cos(rotation.x)*5.0);
    
    
    // Trace rays
    float xx = (2.0 * ((x + 0.5) * invWidth) - 1.0) * angle * aspectratio; 
    float yy = (1.0 - 2.0 * ((y + 0.5) * invHeight)) * angle; 
    vec3 raydir = vec3(xx, yy, -1.0); 
    raydir = normalize(raydir);
    
    
    
    
    //matrix multiplication
    mat3 rotz = mat3(
        cos(rotation.y), -sin(rotation.y), 0, 
        sin(rotation.y), cos(rotation.y), 0, 
        0, 0, 1
    );
    
    mat3 rotx = mat3(
        1,                0,                0,
        0,                cos(rotation.y), -sin(rotation.y), 
        0,                sin(rotation.y),  cos(rotation.y)
    );
        
    mat3 roty = mat3(
        cos(rotation.x),  0, sin(rotation.x), 
        0, 				  1, 0, 
        -sin(rotation.x), 0, cos(rotation.x)
    );
    
    //actually multiplying
    raydir = VecMatMult(rotx, raydir);
    raydir = VecMatMult(roty, raydir);
    
    
    //Material mat, PointLight L, vec3 pos, vec3 normal, vec3 toEye, out vec4 ambient, out vec4 diffuse, out vec4 spec
    //vec3 toEyel = normalize(eye - hitpos);
    
    
    //coloring
    vec3 dist = raymarch(eye, raydir, end);
    vec3 hitpos = eye + (raydir * dist.x);
    vec3 color = vec3(0.0);
    
    vec3 rcolor = getColor(eye, hitpos);
    vec3 acolor = getColor(eye, hitpos);
    
    if(getMat(hitpos).Reflect>0.0) {
    	vec3 normal = estimateNormal(hitpos);
    	//vec3 raydir2 = raydir - 2.0 * dot(raydir, normal) * normal; ;
        vec3 raydir2 = reflect(raydir, normal);
    	vec3 dist2 = raymarch(hitpos, raydir2, end);
    	vec3 hitpos2 = hitpos + (raydir2 * dist2.x);
    
    	//vec3 color = vec3(0.0);
    	//color = litColor.xyz;
    	//vec3 color1 = getColor(eye, hitpos);
    	//vec3 color2 = getColor(eye, hitpos2);
    	//color = (color1*(1.0-getMat(hitpos).Reflect))+(color2*getMat(hitpos).Reflect);
        if(getMat(hitpos2).Reflect>0.0) {
    		vec3 normal2 = estimateNormal(hitpos2);
            vec3 raydir3 = reflect(raydir2, normal2);
    		//vec3 raydir3 = raydir2 - 2.0 * dot(raydir2, normal2) * normal2; ;
    		vec3 dist3 = raymarch(hitpos2, raydir3, end);
    		vec3 hitpos3 = hitpos2 + (raydir3 * dist3.x);
    
    		//vec3 color = vec3(0.0);
    		//color = litColor.xyz;
    		//vec3 acolor = getColor(eye, hitpos2);
    		rcolor = getColor(hitpos2, hitpos3);
    		//color = (color12*(1.0-getMat(hitpos).Reflect))+(color22*getMat(hitpos).Reflect);
        } else {
            //vec3 acolor = getColor(eye, hitpos);
    		rcolor = getColor(hitpos, hitpos2);
            //color = (color1*(1.0-getMat(hitpos).Reflect))+(color2*getMat(hitpos).Reflect);
        }
    }
    //vec3 recolor = (acolor*(1.0-getMat(hitpos).Reflect))+(rcolor*getMat(hitpos).Reflect);
    //vec3 recolor = rcolor;
    vec3 recolor = getColor(eye, hitpos);
    vec3 refcolor = vec3(0.0);
    vec3 refarecolor = vec3(0.0);
    if(getMat(hitpos).ior>0.0) {
        //const vec3 I, const vec3 N, const float ior
        vec3 raydir2 = refract(raydir, estimateNormal(hitpos), 1.0/getMat(hitpos).ior);
        //vec3 raydir2 = 
        vec3 dist2 = mraymarch(hitpos+(raydir*EPSILON), raydir2, end);
        vec3 hitpos2 = hitpos + (raydir2 * dist2.x);
        
        //vec3 raydir3 = refract(raydir2, estimateNormal(hitpos2), getMat(hitpos).ior);
        vec3 dist3 = raymarch(hitpos2+(raydir2*EPSILON), raydir2, end);
        vec3 hitpos3 = hitpos2 + (raydir2 * dist3.x);
        
        refcolor = getColor(eye, hitpos3);
    }
    
    if (getMat(hitpos).Reflect>0.0 && getMat(hitpos).ior>0.0) {
        float kr; 
    	//rSchlick2(estimateNormal(hitpos), eye, 1.0, getMat(hitpos).ior); 
        //float n1, float n2, vec3 normal, vec3 incident, float OBJECT_REFLECTIVITY
        kr = FresnelReflectAmount(1.0, getMat(hitpos).ior, estimateNormal(hitpos), raydir);
    	refarecolor = rcolor * kr + refcolor * (1.0 - kr);
        //refarecolor = rcolor;
        color = refarecolor;
        //color = vec3(1.0) * kr;
    } else if (getMat(hitpos).Reflect>0.0) {
        color = rcolor;
    } else if (getMat(hitpos).ior>0.0) {
        color = refcolor;
    } else {
        color = acolor;
    }
    
    vec2 npos = fragCoord-(iResolution.xy/2.0);
    color -= len(npos.x/(iResolution.x/iResolution.y),npos.y)/iResolution.x*0.2;
    
    //outputting
    fragColor = vec4(color,1);
    
}



