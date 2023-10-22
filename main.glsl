////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
//notes//////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

/*

cooltime = 20.9

*/

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
//data//////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

   // additional bounces //
#define reflection
#define refraction

   // lights //
#define sunLight
//#define skyLight
#define rskyLight
//#define fogSkyLight
//#define inLight


   // post-processing //
#define vignette
#define gamma
#define ao
//#define fog
//#define glow
//#define godRay
//#define bounceGRay
// ^godrays in bounces


//raymarching basic parameters
const float start              =  0.002;
const float end                = 110.0; //also used for fog
const float EPSILON            =  0.001;
const float MIN_EPSILON        =  0.0;
const int   MAX_MARCHING_STEPS = 400;
const float distMultiplier     = 1.0;


//overshoot correction
const float bounceMult = 0.48;

//glow parameters
const float gWidth     = 0.1;
const vec4  gColor     = vec4(1.0, 0.0, 0.0, 1.0);

//godray parameters
const float gMax    = 20.0;
const float gDist   = 0.05;
const float gLength = 12.0;
const float gPower  = 0.1;

//rotation of the camera
vec2 rotation = vec2(0, 0);

//pi
const float M_PI = 3.141592653589;

////////////////////////////////////////////////////////////////////////////////////////////

//Lights

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
    vec4(0.6f, 0.6f, 0.6f, 1.0),
    vec4(0.6f, 0.6f, 0.6f, 1.0),
    vec4(0.6f, 0.6f, 0.6f, 1.0),
    vec3(0.0,10.0,0.0),
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


_sunLight = DirectionalLight(
    vec4((vec3(222, 214, 171)/255.0)*0.1, 1.0),
    vec4((vec3(222, 214, 171)/255.0)*3.80, 1.0),
    vec4(vec3(222, 214, 171)/255.0, 1.0),
    normalize(vec3(-0.5, -0.4, -0.6)),
    1.0
);

DirectionalLight _skyLight = DirectionalLight(
    vec4((vec3(255, 255, 255)/255.0)*0.6, 1.0),
    vec4((vec3(255, 255, 255)/255.0)*4.55, 1.0),
    vec4((vec3(255, 255, 255)/255.0), 1.0),
    normalize(vec3(0.3f, -1.0f, 0.1f)),
    1.0
);

DirectionalLight _rskyLight = DirectionalLight(
    vec4((vec3(102, 153, 255)/255.0), 1.0),
    vec4((vec3(102, 153, 255)/255.0)*3.55, 1.0),
    vec4((vec3(191, 181, 125)/255.0), 1.0),
    normalize(vec3(0.0f, -1.0f, 0.0f)),
    1.0
);

DirectionalLight _fogSkyLight = DirectionalLight(
    vec4((vec3(117, 139, 189)/255.0), 1.0),
    vec4((vec3(117, 139, 189)/255.0)*0.55, 1.0),
    vec4((vec3(117, 139, 189)/255.0), 1.0),
    normalize(vec3(0.0f, -1.0f, 0.0f)),
    1.0
);

DirectionalLight _inLight = DirectionalLight(
    vec4(vec3(0), 1.0),
    vec4(vec3(1.30,1.00,0.70)*0.95, 1.0),
    vec4(1.30,1.00,0.70, 1.0),
    normalize(vec3(0.5, 0.0, 0.6)),
    1.0
);

////////////////////////////////////////////////////////////////////////////////////////////

//Materials

struct Material
{
    vec4 Ambient;
    vec4 Diffuse;
    bool matte;
    vec4 Specular; // .w = SpecPower
    float Refract;
    float ior;
};
  
Material _landMaterial = Material(
    vec4(0.48f, 0.77f, 0.46f, 1.0),
    vec4(0.48f, 0.77f, 0.46f, 1.0),
    false,
    vec4(vec3(1.0), 20.0),
    0.0,
    1.46
);

Material _plainMaterial = Material(
    vec4(0.30f, 0.30f, 0.30f, 0.46f),
    vec4(0.30f, 0.30f, 0.30f, 0.46f),
    false,
    vec4(vec3(1.0), 100.0),
    0.0,
    1.46
);


Material _darkFloorMaterial = Material(
    vec4(0.10f, 0.10f, 0.10f, 0.46f),
    vec4(0.10f, 0.10f, 0.10f, 0.46f),
    false,
    vec4(vec3(0.7), 20.0),
    0.0,
    3.0
);

Material _FloorMaterial = Material(
    vec4(vec3(0.15f), 0.46f),
    vec4(vec3(0.15f), 0.46f),
    false,
    vec4(vec3(0.7), 20.0),
    0.0,
    4.0
);


Material _darkPlainMaterial = Material(
    vec4(0.10f, 0.10f, 0.10f, 0.46f),
    vec4(0.10f, 0.10f, 0.10f, 0.46f),
    false,
    vec4(vec3(1.0), 20.0),
    0.0,
    1.46
);

Material _reflectiveMaterial = Material(
    vec4(0.05f, 0.05f, 0.05f, 0.46f),
    vec4(0.05f, 0.05f, 0.05f, 0.46f),
    false,
    vec4(vec3(1.0), 20.0),
    0.0,
    100.0
);
Material _glassMaterial = Material(
    vec4(vec3(0.3, 0.3, 0.5), 0.46f),
    vec4(vec3(0.3, 0.3, 0.5), 0.46f),
    false,
    vec4(vec3(1.0), 20.0),
    0.98,
    1.5
);

Material _wavesMaterial = Material(
    vec4(0.137f, 0.42f, 0.556f, 1.0),
    vec4(0.137f, 0.42f, 0.556f, 1.0),
    false,
    vec4(vec3(1.0), 20.0),
    0.9,
    1.3
);
/*
Material _skyMaterial = Material(
    vec4((vec3(43)/255.0), 1.0),
    vec4((vec3(43)/255.0), 1.0),
    true,
    vec4(vec3(1.0), 20.0),
    0.0,
    0.0
);
*/
Material _skyMaterial = Material(
    vec4((vec3(102, 153, 255)/255.0), 1.0),
    vec4((vec3(102, 153, 255)/255.0)*1.3, 1.0),
    true,
    vec4(vec3(1.0), 20.0),
    0.0,
    0.0
);
Material _sunMaterial = Material(
    vec4(vec3(222, 214, 171)/255.0*2.0, 1.0),
    vec4(vec3(222, 214, 171)/255.0*0.0, 1.0),
    true,
    vec4(vec3(1.0), 20.0),
    0.0,
    0.0
);

Material _fogSkyMaterial = Material(
    vec4((vec3(117, 139, 189)/255.0), 1.0),
    vec4((vec3(117, 139, 189)/255.0)*1.3, 1.0),
    true,
    vec4(vec3(1.0), 20.0),
    0.0,
    0.0
);

Material _redMaterial = Material(
    vec4((vec3(181, 23, 23)/255.0), 1.0),
    vec4((vec3(181, 23, 23)/255.0), 1.0),
    false,
    vec4((vec3(181, 23, 23)/255.0), 40.0),
    0.0,
    1.4
);

Material _fracIMaterial = Material(
    vec4(0.7594257,0.57437592,0.582409234,0.579345452),
    vec4((vec3(181, 23, 23)/255.0),1.0),
    true,
    vec4(0.7594257,0.57437592,0.582409234,0.579345452),
    0.0,
    0.0
);


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
//functions/////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

//utility functions


////////////////////////////////////////////////////////////////////////////////////////////

//Matricies

//multiplies a matrix and vector
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

//rotation around the x line
mat3 matRotX(float rot) {
    return mat3(
        1,                0,                0,
        0,                cos(rot), -sin(rot), 
        0,                sin(rot),  cos(rot)
    );
}

//rotation around the y line
mat3 matRotY(float rot) {
    return mat3(
        cos(rot),  0, sin(rot), 
        0, 				  1, 0, 
        -sin(rot), 0, cos(rot)
    );
}

//rotation around the z line
mat3 matRotZ(float rot) {
   return mat3(
        1,                0,                0,
        0,                cos(rot), -sin(rot), 
        0,                sin(rot),  cos(rot)
    ); 
}

////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

//https://iquilezles.org/articles/distfunctions
//opSmoothUnion
float osp( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); 
}

//https://iquilezles.org/articles/distfunctions
//opSmoothSubtraction
float osm( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); 
}

//https://iquilezles.org/articles/distfunctions
//opSmoothIntersection
float osx( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h); 
}


//distance functions


//https://iquilezles.org/articles/distfunctions
//distance function for box
float sdBox( vec3 p, vec3 b ) {
  vec3 d = abs(p) - b;
  return length(max(d,0.0))
         + min(max(d.x,max(d.y,d.z)),0.0); // remove this line for an only partially signed sdf 
}

//https://iquilezles.org/articles/distfunctions
//distance function for sphere
float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}

//https://iquilezles.org/articles/distfunctions
//distance function for plane
float sdPlane( vec3 p, vec3 n ) {
  // n must be normalized
  
  //return max(dot(p,n),dot((p*-n)+(-n*EPSILON),-n));
  return dot(p,n);
}

//http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
float sdMandelbulb(vec3 pos) {
	vec3 z = pos;
	float dr = 1.0;
	float r = 0.0;
    
    int Iterations = 7;
    float Bailout = 200.0;
    float Power = 6.3+sin(34.65*0.5)*2.0;//usually 6.3+sin(iTime*0.5)*2.0
    
	for (int i = 0; i < Iterations ; i++) {
		r = length(z);
		if (r>Bailout) break;
		
		// convert to polar coordinates
		float theta = acos(z.z/r);
		float phi = atan(z.y,z.x);
		dr =  pow( r, Power-1.0)*Power*dr + 1.0;
		
		// scale and rotate the point
		float zr = pow( r,Power);
		theta = theta*Power;
		phi = phi*Power;
		
		// convert back to cartesian coordinates
		z = zr*vec3(sin(theta)*cos(phi), sin(phi)*sin(theta), cos(theta));
		z+=pos;
	}
	return 0.5*log(r)*r/dr;
}

float sdMengerBound ( in vec3 p ) {
    
    float s = 1.0;
    vec3 a = mod( p*s, 2.0 )-1.0;
    s *= 3.0;
    vec3 r = abs(1.0 - 3.0*abs(a));

    float da = max(r.x,r.y);
    float db = max(r.y,r.z);
    float dc = max(r.z,r.x);
    float c = (min(da,min(db,dc))-1.0)/s;
    
    return max(sdBox(p, vec3(1)), c);
}

//https://iquilezles.org/articles/menger
float sdMengerSponge( in vec3 p ) {
   //float d = sdMengerBound(p);
   float d = sdBox(p, vec3(1));
   if(d>EPSILON) return d; //bounding box (1 iteration of a menger sponge)

   float s = 1.0;
   for( int m=0; m<5; m++ )
   {
       /*
       p = VecMatMult(matRotX(0.1), p);
       p = VecMatMult(matRotY(0.5), p);
       p = VecMatMult(matRotZ(-0.2), p);
       */
       vec3 a = mod( p*s, 2.0 )-1.0;
       s *= 3.0;
       vec3 r = abs(1.0 - 3.0*abs(a));

       float da = max(r.x,r.y);
       float db = max(r.y,r.z);
       float dc = max(r.z,r.x);
       float c = (min(da,min(db,dc))-1.0)/s;

      d = max(d,c);
   }

   return d;
}

vec2 hash( vec2 x, float a, float b ) {
    vec2 k = vec2( a, b );
    x = x*k + k.yx;
    return -1.0 + 2.0*fract( 16.0 * k*fract( x.x*x.y*(x.x+x.y)) );
} 

float nois( in vec2 p, float a, float b ) {
    vec2 i = floor( p );
    vec2 f = fract( p );
	
	vec2 u = f*f*(3.0-2.0*f);

    return mix( mix( dot( hash( i + vec2(0.0,0.0), a, b ), f - vec2(0.0,0.0) ), 
                     dot( hash( i + vec2(1.0,0.0), a, b ), f - vec2(1.0,0.0) ), u.x),
                mix( dot( hash( i + vec2(0.0,1.0), a, b ), f - vec2(0.0,1.0) ), 
                     dot( hash( i + vec2(1.0,1.0), a, b ), f - vec2(1.0,1.0) ), u.x), u.y);
}
float nois( in vec2 p ) {
    return nois(p, 0.529243235, 0.46198324);
}

//https://www.shadertoy.com/view/4lfBWB (WaveNF and WaveHT)

float WaveHt (vec2 p)
{
  float tCur = iTime;
  float tWav = 0.2 * tCur;
  
  mat2 qRot = mat2 (0.8, -0.6, 0.6, 0.8);
  vec4 t4, v4;
  vec2 q, t, tw;
  float wFreq, wAmp, h;
  q = 0.5 * p + vec2 (0., tCur);
  h = 0.6 * sin (dot (q, vec2 (-0.05, 1.))) + 0.45 * sin (dot (q, vec2 (0.1, 1.2))) +
     0.3 * sin (dot (q, vec2 (-0.2, 1.4)));
  q = p;
  wFreq = 1.;
  wAmp = 1.;
  tw = tWav * vec2 (1., -1.);
  for (int j = 0; j < 3; j ++) {
    q *= qRot;
    t4 = q.xyxy * wFreq + tw.xxyy;
    t = vec2 (nois (t4.xy), nois (t4.zw));
    t4 += 2. * t.xxyy - 1.;
    v4 = (1. - abs (sin (t4))) * (abs (sin (t4)) + abs (cos (t4)));
    t = 1. - sqrt (v4.xz * v4.yw);
    t *= t;
    t *= t;
    h += wAmp * dot (t, t);
    wFreq *= 2.;
    wAmp *= 0.5;
  }
  return h;
}
vec3 WaveNoise (vec3 p, float d)
{
  vec3 vn;
  vec2 e;
  e = vec2 (max (0.01, 0.005 * d * d), 0.);
  p *= 0.5;
  vn.xz = 0.5 * (WaveHt (p.xz) - vec2 (WaveHt (p.xz + e.xy),  WaveHt (p.xz + e.yx)));
  vn.y = e.x;
  return normalize (vn);
}

float checkers( in vec3 p ) {
    vec3 s = sign(fract(p*.5)-.5);
    return .5 - .5*s.x*s.y*s.z;
}

float opaqueSDF(vec3 point) {
    float dist = sdSphere(point - vec3(0,1.0+sin(iTime),0),1.0);
    dist = min(dist, sdSphere(point - vec3(3,0,0),1.0));
    dist = min(dist, sdBox(point - vec3(0,0,-5),vec3(7,1.5,0.5)));
    dist = min(dist, sdPlane(point - vec3(0,-2,0),vec3(0,1,0)));
    dist = min(dist, sdMengerSponge(point - vec3(sin(iTime*0.648)*2.0,4,0)));
    return dist;
}


//The scene standard distance function
float sceneSDF(vec3 point) {
    return
        min(
            opaqueSDF(point),
            sdBox(point - vec3(-3,0,0),vec3(1.0, 1.0, 0.1))
            );                    
}
////////////////////////////////////////////////////////////////////////////////////////////

//getting the material of a point

Material getMat(vec3 point) {
    if(sdBox(point - vec3(-3,0,0),vec3(1.0, 1.0, 0.1))<=EPSILON) {return _glassMaterial;}
    if(sdSphere(point - vec3(0,1.0+sin(iTime),0),1.0)<=EPSILON) {return _reflectiveMaterial;}
    if(sdSphere(point - vec3(3,0,0),1.0)<=EPSILON) {return _redMaterial;}
    if(sdBox(point - vec3(0,0,-5),vec3(7,1.5,0.5))<=EPSILON) {return _plainMaterial;}
    if(sdBox(point - vec3(sin(iTime*0.648)*2.0,4,0), vec3(1))<=EPSILON) {return _redMaterial;} //mengerSponge
    if(sdPlane(point - vec3(0,-2,0),vec3(0,1,0))<=EPSILON) {
        if(checkers(point)>0.5) {
        	return _FloorMaterial;
        } else {
            return _darkFloorMaterial;
        }
    }
    if(sdSphere(point+vec3(-0.5, -0.4, -0.6)*end, 3.0)<EPSILON) return _sunMaterial;
    return _skyMaterial;
}

////////////////////////////////////////////////////////////////////////////////////////////

//raymarching


//raymarch algorithm
float raymarch(vec3 eye, vec3 viewRayDirection, float endd) {
    viewRayDirection = normalize(viewRayDirection);
    float depth = start;
    
    //float rimcheck = end;
    bool bouncecheck = false;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + (depth * viewRayDirection));
        dist /= distMultiplier;
        if (dist < EPSILON && dist >= 0.0) {        
            return depth;
        }
        
        if(bouncecheck) {
            dist *= bounceMult;
        }
        
        if(dist < EPSILON) {
            bouncecheck = true;
        } else {
            bouncecheck = false;
        }
        
        //Move along the view ray
        depth += dist;
        
        if (depth >= min(endd, end) )  {
            // Gone too far; give up
            return endd;
        }
    }
    return 0.0;
    
}

float opaquemarch(vec3 eye, vec3 viewRayDirection, float endd) {
    viewRayDirection = normalize(viewRayDirection);
    float depth = start;
    
    //float rimcheck = end;
    bool bouncecheck = false;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = opaqueSDF(eye + (depth * viewRayDirection));
        dist /= distMultiplier;
        if (dist < EPSILON && dist >= 0.0) {        
            return depth;
        }
        
        if(bouncecheck) {
            dist *= bounceMult;
        }
        
        if(dist < EPSILON) {
            bouncecheck = true;
        } else {
            bouncecheck = false;
        }
        
        //Move along the view ray
        depth += dist;
        
        if (depth >= min(endd, end) )  {
            // Gone too far; give up
            return endd;
        }
    }
    return 0.0;
    
}

//gives additional facts about the ray, distance, iteration count, and the minumum distance
vec3 iraymarch(vec3 eye, vec3 viewRayDirection, float endd) {
    viewRayDirection = normalize(viewRayDirection);
    float depth = start;
    
    //float rimcheck = end;
    bool bouncecheck = false;
    
    float mindist = endd;

    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + (depth * viewRayDirection));
        dist /= distMultiplier;
        
        if(depth > gWidth*5.0) { mindist = min(mindist,dist); }
        
        if (dist < EPSILON && dist > MIN_EPSILON) {        
            return vec3(depth,i,mindist);
        }
        if(bouncecheck) {
            dist *= 0.42;
        }
        
        if(dist < EPSILON) {
            bouncecheck = true;
        } else {
            bouncecheck = false;
        }
        
        //Move along the view ray
        depth += dist;
        
        
        
        if (depth >= min(endd, end) )  {
            // Gone too far; give up
            return vec3(endd,MAX_MARCHING_STEPS,mindist);
        }
    }
    return vec3(endd,MAX_MARCHING_STEPS,mindist);
    
}


//raycasting until it hits a different material
float mraycast(vec3 eye, vec3 viewRayDirection, float endd) {
    viewRayDirection = normalize(viewRayDirection);
    float returning = 0.0;
    float depth = start;
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
    	    //returning = depth;
            return depth;
            //break;
    	}
    	// Move along the view ray
	
    	if (depth >= endd || depth >= end) {
    	    // Gone too far; give up
    	    //returning = endd;
            //break;
            return endd;
    	}
        
        //if (i+1 == MAX_MARCHING_STEPS) {returning.z = 1.0;}
	}
    return returning;
	
}


//get the ray from screen coordinates
vec3 getRayFromCoords(vec2 point) {
    float width = iResolution.x, height = iResolution.y; 
	float invWidth = 1.0 / float(width), invHeight = 1.0 / float(height); 
	float fov = -90.0*(M_PI/180.0), aspectratio = float(width) / float(height); 
	float angle = tan(fov/2.0);
    //float angle = tan(M_PI * 0.5 * fov); 
    //angle = 0.0;
    //angle = fov/2.0;
    
    float x = (2.0 * ((point.x + 0.5) * invWidth) - 1.0) * angle * aspectratio; 
    float y = (1.0 - 2.0 * ((point.y + 0.5) * invHeight)) * angle; 
    vec3 outt = vec3(x, y, -1.0); 
    outt = normalize(outt);
    
    //return vec3(p,2.0);
    return outt;
}

////////////////////////////////////////////////////////////////////////////////////////////

//graphics, colors, etc

//  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  / //

//lights

//compute the phong light coefficients of a point light from the material and the light (yaay long words)
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


//compute the phong light coefficients of a directional light from the material and the light (yaay long words)
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
    //ambient = vec4(distance(vec3(0.0), ambient.xyz));

    // Add diffuse and specular term, provided the surface is in 
    // the line of site of the light.
    
    float diffuseFactor = dot(lightVec, normal);
    //diffuseFactor = 0.5;

    // Flatten to avoid dynamic branching.
    if( diffuseFactor > 0.0f )
    {
        vec3 v         = reflect(-lightVec, normal);
        float specFactor = pow(max(dot(v, toEye), 0.0f), mat.Specular.w);
                    
        diffuse = diffuseFactor * mat.Diffuse * L.Diffuse;
        
        //diffuse = mat.Diffuse * L.Diffuse;
        spec    = specFactor * mat.Specular * L.Specular;
    }
    //diffuse = mat.Diffuse * L.Diffuse;
    //diffuse = vec4(distance(vec3(0.0), diffuse.xyz));
    //spec = vec4(distance(vec3(0.0), spec.xyz));
}


vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}



//  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  / //

//refraction amount

float Fresnel (float n1, float n2, vec3 normal, vec3 incident) {
    float r0 = (n1-n2) / (n1+n2);
    r0 *= r0;
    
    float costh = -dot(normal, incident);
    float a = 1.0 - costh;
    
    return r0 + ((1.0 - r0)*a*a*a*a*a);
}

//  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  / //

//ambient occlusion

vec4 calcAO(vec3 point, vec3 normal) {
    vec4 totao = vec4(0.0);
    float sca = 1.0;

    for (int aoi = 0; aoi < 16; aoi++)
    {
        float hr = 0.01 + 0.02 * float(aoi * aoi);
        vec3 aopos = point + normal * hr;
        float dd = sceneSDF(aopos);
        float aoo = clamp(-(dd - hr), 0.0, 1.0);
        totao += aoo * sca * vec4(1.0, 1.0, 1.0, 1.0);
        sca *= 0.75;
    }

    const float aoCoef = 0.5;
    totao.w = 1.0 - clamp(aoCoef * totao.w, 0.0, 1.0);

    return totao;
}

//shadows

//https://iquilezles.org/articles/rmshadows
float softshadow( vec3 eye, vec3 to, float endd, float k ) {
    vec3 ray = normalize(to - eye);
    
    float res = 1.0;
    for( float t=start; t<endd; )
    {
        float h = opaqueSDF(eye + ray*t);
        if( h<EPSILON )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

//  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  /  / //

//getting phong reflection color, split into base and specular

mat2x4 getMatColorFromPoint(Material mat, vec3 normal, vec3 from, vec3 point) {
    //return estimateNormal(point);
    
    mat2x4 color = mat2x4(0);
    
    vec4 ambient = vec4(0);
    vec4 diffuse = vec4(0);
    vec4 spec = vec4(0);
    
    vec4 A, D, S;
    
    vec3 toEyel = normalize(from - point);
    //Material mat = getMat(point);
    
    //ComputePointLight(mat,_pointLight,point,estimateNormal(point),toEyel, A, D, S);
    //ambient += A;
    //if(!hitsObject(point,_pointLight.Position)) {
    //	diffuse += D;
    //	spec    += S;
    //}
    //float shadow = getShadow(from, point, _pointLight.Position);
    //diffuse += D * shadow;
    //spec    += S * shadow;
    float shadow = 0.0;
    #ifdef sunLight
    ComputeDirectionalLight(mat,_sunLight,normal,toEyel, A, D, S);
    ambient += A;
    //float shadow = getShadow(from, point, -_dirLight.Direction*end);
    shadow = softshadow(point, -_sunLight.Direction*end, end, 8.0);
    //shadow = 1.0;
    //shadow *= 2.0;
    diffuse += D * shadow;
    spec    += S * shadow;
    #endif
    #ifdef skyLight
    ComputeDirectionalLight(mat,_skyLight,normal,toEyel,A,D,S);
    ambient+=A;
    shadow = softshadow(point, -_skyLight.Direction*end, end, 16.0);
    //shadow = 1.0;
    diffuse+=D * shadow;
    spec+=S * shadow;
    //diffuse+=D * shadow;
    //spec+=S * shadow;
    #endif
    #ifdef rskyLight
    ComputeDirectionalLight(mat,_rskyLight,normal,toEyel,A,D,S);
    ambient+=A;
    shadow = softshadow(point, -_rskyLight.Direction*end, end, 16.0);
    //shadow = 1.0;
    diffuse+=D * shadow;
    spec+=S * shadow;
    //diffuse+=D * shadow;
    //spec+=S * shadow;
    #endif
    #ifdef fogSkyLight
    ComputeDirectionalLight(mat,_fogSkyLight,normal,toEyel,A,D,S);
    ambient+=A;
    shadow = softshadow(point, -_fogSkyLight.Direction*end, end, 16.0);
    //shadow = 1.0;
    diffuse+=D * shadow;
    spec+=S * shadow;
    //diffuse+=D * shadow;
    //spec+=S * shadow;
    #endif
    #ifdef inLight
    ComputeDirectionalLight(mat,_inLight,normal,toEyel,A,D,S);
    ambient+=A;
    shadow = softshadow(point, -_inLight.Direction*end, end, 4.0);
    //shadow = 1.0;
    diffuse+=D * shadow;
    spec+=S * shadow;
    #endif
	
#ifdef ao
    vec4 matao = calcAO(point, normal);
#else
    vec4 matao = vec4(0);
#endif
    //return mat2x4(vec4(vec3(ao),1),vec4(0,0,0,1));
    //ao = vec4(1.0);
    matao = 1.0-matao;
    //ambient = vec4(0);
    if(!mat.matte) {
        
        color = mat2x4(ambient*matao+diffuse,spec);
        //color = spec;
        //color = mat.Ambient*2.0*ao;
        //color = vec4(estimateNormal(point), 1.0);
    } else {
        color = mat2x4(mat.Diffuse,vec4(0));
    }
    
    //return color.xyz;
    return color;
    
}

//basic stuff
void doMarch(vec3 eye, vec3 ray, out vec3 march, out float len, out vec3 point, out vec3 surfacepoint ) {
    march = iraymarch(eye,ray,end);
    len = march.x;
    point = eye + (ray * (len));
    surfacepoint = point-1.0*EPSILON*ray;
}
void doColor(Material mat,vec3 eye,vec3 surfacepoint,vec3 march, out vec4 col,out vec3 normal,out mat2x4 matColor,out vec4 spec) {
    col = vec4(0);
    normal = estimateNormal(surfacepoint);
    matColor = getMatColorFromPoint(mat, normal, eye, surfacepoint);
    spec = matColor[1];
    
    //iteration material
    if(mat == _fracIMaterial) { col = (march.y / float(MAX_MARCHING_STEPS))*mat.Diffuse; }
    else                      { col = matColor[0]; }
    //col = vec4(normal, 1.0);
    //col = spec;
    //col = Fresnel(1.0, mat.ior, normal, ray);
}

//extra bounces
void getRefr(vec4 col, vec3 point,vec3 ray,vec3 normal,Material mat, out vec3 rrEye, out vec3 rrRay) {
    vec3 refractRay = refract(ray, normal, 1.0/mat.ior);
    float refractDist = mraycast(point, refractRay, end);
    vec3 refractExit = point + refractRay * refractDist;

    rrEye = refractExit;
    rrRay = ray;
    //return mix(col, getColorFromRay2(refractExit, ray), mat.Refract);
    
}
void getRefl(vec3 point,vec3 ray,vec3 normal, Material mat, out float fresnel, out vec3 rlEye, out vec3 rlRay) {
    if(!mat.matte) {
        rlEye = point;
        rlRay = reflect(ray, normal);
    	//ref = getColorFromRay2(point, reflect(ray, normal));
    	fresnel = clamp(Fresnel(1.0, mat.ior, normal, ray), 0.0, 1.0);
    }
    //return ref;
}

//special effects
vec4 getGlow(vec3 march) {
    /*
    float mindist = march.z;
    mindist *= 1.0/gWidth;
    if(march.z >= EPSILON && mindist <= 1.0) {
        return (1.0-mindist)*gColor;
    }
    return vec4(0.0);
*/
    return march.y*gColor;
    
}
vec4 getGRay(vec3 eye,vec3 ray, float len) {
    float gAmount = 0.0;
    float gDepth  = 0.0; 
    
    int i = 0;
    for(; i<int(gMax/gDist); i++) {
        gDepth += gDist;
        if(gDepth >= len) break;
        
        float gAtPoint = opaquemarch(eye+ray*gDepth, -_skyLight.Direction, gLength)/gLength;
        gAmount += gAtPoint==1.0 ? 1.0 : 0.0;
    }
    //col += (i==0) ? vec4(0.0) : (gAmount/float(i))*_skyLight.Diffuse*gPower;
    return (gAmount/floor(gMax/gDist))*_skyLight.Diffuse*gPower;
}

vec4 getColorFromRay4(vec3 eye, vec3 ray) {
    ray = normalize(ray);
    
    vec3 col = vec3(0,0,0);
    float len = raymarch(eye,ray,end);
    vec3 point = eye + (ray * len);
    
    //return vec3(1.0);
    return getMatColorFromPoint(getMat(point), estimateNormal(point), eye, point)[0];
}


vec4 getColorFromRay3(vec3 eye, vec3 ray) {
    ray = normalize(ray);
    
    // marching //
    vec3 march,point,surfacepoint;
    float len;
    doMarch(eye,ray, march,len,point,surfacepoint );
    
    // material //
    Material mat = getMat(point);
    
    
    // coloring //
    vec4 col, spec;
    vec3 normal;
    mat2x4 matColor;
    doColor(mat,eye,surfacepoint,march, col,normal,matColor,spec);
    
    
#ifdef refraction
    if(mat.Refract > 0.0) {
        vec3 rrEye, rrRay;
    	getRefr(col, point, ray, normal, mat, rrEye, rrRay);
    	col = mix(col, getColorFromRay4(rrEye, rrRay), mat.Refract);
    }
#endif
    
    vec4 ref = vec4(0);
    float fresnel = 0.0;
#ifdef reflection
    if(!mat.matte) {
    	vec3 rlEye, rlRay;
    	getRefl(point, ray, normal, mat, fresnel, rlEye, rlRay);
    	ref = getColorFromRay4(rlEye, rlRay);
    }
#endif
    
#ifdef bounceGRay
    col += getGRay(eye, ray, len);
#endif
    
#ifdef glow
    col += getGlow(march);
#endif
    
    
    col += mix(spec,ref,fresnel);
    
    return col;
    
    
}


vec4 getColorFromRay2(vec3 eye, vec3 ray) {
    ray = normalize(ray);
    //return vec4(ray, 1.0);
    
    // marching //
    vec3 march,point,surfacepoint;
    float len;
    doMarch(eye,ray, march,len,point,surfacepoint );
    
    // material //
    Material mat = getMat(point);
    
    
    // coloring //
    vec4 col, spec;
    vec3 normal;
    mat2x4 matColor;
    doColor(mat,eye,surfacepoint,march, col,normal,matColor,spec);
    
    
#ifdef refraction
    if(mat.Refract > 0.0) {
        vec3 rrEye, rrRay;
    	getRefr(col, point, ray, normal, mat, rrEye, rrRay);
    	col = mix(col, getColorFromRay3(rrEye, rrRay), mat.Refract);
    }
#endif
    
    vec4 ref = vec4(0);
    float fresnel = 0.0;
#ifdef reflection
    if(!mat.matte) {
    	vec3 rlEye, rlRay;
    	getRefl(point, ray, normal, mat, fresnel, rlEye, rlRay);
    	ref = getColorFromRay4(rlEye, rlRay);
    }
#endif
    
#ifdef bounceGRay
    col += getGRay(eye, ray, len);
#endif
    
#ifdef glow
    col += getGlow(march);
#endif
    
    
    col += mix(spec,ref,fresnel);
    
    return col;
    
    
}

vec4 getColorFromRay(vec3 eye, vec3 ray) {
    ray = normalize(ray);
    
    // marching //
    vec3 march,point,surfacepoint;
    float len;
    doMarch(eye,ray, march,len,point,surfacepoint );
    //return vec4(vec3(len), 1.0);
    
    // material //
    Material mat = getMat(point);
    
    
    // coloring //
    vec4 col, spec;
    vec3 normal;
    //return vec4(normal, 1.0);
    mat2x4 matColor;
    doColor(mat,eye,surfacepoint,march,col,normal,matColor,spec);
    //col = vec4(vec3(Fresnel(1.0, mat.ior, normal, ray)),1.0);
    
    
    
#ifdef refraction
    if(mat.Refract > 0.0) {
        vec3 rrEye, rrRay;
    	getRefr(col, point, ray, normal, mat, rrEye, rrRay);
    	col = mix(col, getColorFromRay2(rrEye, rrRay), mat.Refract);
    }
#endif
    
    vec4 ref = vec4(0);
    float fresnel = 0.0;
#ifdef reflection
    if(!mat.matte) {
    	vec3 rlEye, rlRay;
    	getRefl(point, ray, normal, mat, fresnel, rlEye, rlRay);
    	ref = getColorFromRay2(rlEye, rlRay);
    }
#endif
    
#ifdef godRay
    col += getGRay(eye, ray, len);
#endif
    
#ifdef glow
    col += getGlow(march);
#endif
    
    col += mix(spec,ref,fresnel);
    
    return col;
    
    
}

//Note: the above functions that start with "getColorFromRay" all are copies used to add reflections

//Post Proccesing


//https://iquilezles.org/articles/fog
vec3 applyFog( in vec3  rgb,      // original color of the pixel
               in float dist, // camera to point distance
               in float density,
               in float c,
               in vec3  rayDir,   // camera to point vector
               in vec3  sunDir )  // sun light direction
{
    float fogAmount = c * (1.0 - exp( -dist*density ));
    float sunAmount = max( dot( rayDir, sunDir ), 0.0 );
    #ifdef sunlight
    vec3  fogColor  = mix( normalize(_skyLight.Diffuse.xyz), // bluish
                           normalize(_sunLight.Diffuse.xyz), // yellowish
                           pow(sunAmount,8.0) );
    #else
    vec3 fogColor = normalize(_skyLight.Diffuse.xyz);
    #endif
    return mix( rgb, fogColor, fogAmount );
}

////////////////////////////////////////////////////////////////////////////////////////////





//Main image function


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    
//emergency use
    /*
    if(iTime > 5.0) {
        fragColor = vec4(1.0);
        return;
    }
*/

    //rotation.x = sin(iTime)/1.0+3.14;
    //rotation.x = M_PI+M_PI/4.0;
	//rotation.y = sin(iTime*3.0)/3.0;
    //rotation.x = 2.0*M_PI+sin(iTime);
    float speed = 0.5;
    //rotation.x = iTime*speed;
    //rotation.x = 20.9*speed;
    //rotation.x = (M_PI+3.0)+M_PI/2.0-((iMouse.x-30.0)/iResolution.x)*M_PI;
	//rotation.y = -30.0;
    //rotation.x = (180.0+50.0)*(M_PI/180.0);
    rotation.y = 0.0;
    rotation.y *= M_PI/180.0;
	
	//vec3 eye = vec3(sin(rotation.x)*5.0,-sin(iTime*3.0)*2.0,cos(rotation.x)*5.0);
	//vec3 eye = vec3(sin(iTime*speed)*5.0,1,cos(iTime*speed)*5.0);
    
    //vec3 eye = vec3(sin(rotation.x)*0.8,0,cos(rotation.x)*0.8);
    vec3 eye = vec3(sin(rotation.x)*10.0,1,cos(rotation.x)*10.0);
    
    //vec3 eye = vec3(1.5,0,1.5);
    //vec3 eye = vec3(-5,0,5);
    
    
    vec3 raydir = getRayFromCoords(fragCoord);
    raydir = normalize(raydir);  
    raydir = VecMatMult(matRotX(rotation.y), raydir);
    raydir = VecMatMult(matRotY(rotation.x), raydir);
    vec3 col = getColorFromRay(eye,raydir).xyz;
    
#ifdef fog
    col = applyFog(col, raymarch(eye, raydir, end), 1.0/(end/5.0), 0.9, raydir, -_sunLight.Direction);
#endif
    
#ifdef gamma
    col = pow( col, vec3(1.0/2.2) );
#endif
    
#ifdef vignette
    vec2 npos = fragCoord-(iResolution.xy/2.0);
    col -= length(vec2(npos.x/(iResolution.x/iResolution.y),npos.y))/iResolution.x*0.5;
#endif
    
    
    // Output to screen
    fragColor = vec4(col,1.0);
}








