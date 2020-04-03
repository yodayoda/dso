/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#include <iostream>
#include <string>
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"



#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class SampleOutputWrapper : public Output3DWrapper
{
public:
      inline SampleOutputWrapper() : numPCL(0),
				     isSavePCL(true),
	plyFileName("/tmp/dso.ply"),
	odometryFileName("/tmp/odometry.txt")
        {
	plyFile.open(plyFileName);
	plyFile << "ply\nformat ascii 1.0\nelement vertex 000000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
	odometryFile.open(odometryFileName);
	odometryFile << "x y z\n";
            printf("OUT: Created SampleOutputWrapper\n");
        }

        virtual ~SampleOutputWrapper()
        {
	if (plyFile.is_open())
	  {
	    plyFile.seekp(0, std::ios::beg);
	    plyFile << "ply\nformat ascii 1.0\nelement vertex " << std::setfill('0') << std::setw(9) << numPCL << std::endl;
	    plyFile.flush();
	    plyFile.close();

	    std::cout << "Written flushed file to disk" << std::endl;
	  }
	if (odometryFile.is_open())
	  {
	    odometryFile.flush();
	    odometryFile.close();

	    std::cout << "Written flushed file to disk" << std::endl;
	  }

            printf("OUT: Destroyed SampleOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
	// printf("OUT: got graph with %d edges\n", (int)connectivity.size());

	// int maxWrite = 5;

	// for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
	//   {
	//     int idHost = p.first>>32;
	//     int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
	//     printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
	//     maxWrite--;
	//     if(maxWrite==0) break;
	//   }
        }



        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
      {
	float fx, fy, cx, cy;
	float fxi, fyi, cxi, cyi;
	//float colorIntensity = 1.0f;
	fx = HCalib->fxl();
	fy = HCalib->fyl();
	cx = HCalib->cxl();
	cy = HCalib->cyl();
	fxi = 1 / fx;
	fyi = 1 / fy;
	cxi = -cx / fx;
	cyi = -cy / fy;

	if (not final)// or not final)
        {
            for(FrameHessian* f : frames)
            {
		if (f->shell->poseValid)
		  {
		    auto const& m = f->shell->camToWorld.matrix3x4();

		    // use only marginalized points.
		    auto const& points = f->pointHessiansMarginalized;


		    for (auto const* p : points)
                {
			float depth = 1.0f / p->idepth;
			auto const x = (p->u * fxi + cxi) * depth;
			auto const y = (p->v * fyi + cyi) * depth;
			auto const z = depth * (1 + 2 * fxi);

			Eigen::Vector4d camPoint(x, y, z, 1.f);
			Eigen::Vector3d worldPoint = m * camPoint;

			if (isSavePCL && plyFile.is_open())
			  {
			    plyFile << worldPoint[0] << " " << worldPoint[1] << " " << worldPoint[2] << "\n";
			    numPCL++;
			  }
		      }
		  }
	      }
	    if(isSavePCL && plyFile.is_open()) {
	      std::cout << "\n\n\nWritten " << numPCL << " points to file " << plyFileName << std::endl;
	      plyFile.flush();

                }
            }
        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
            printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
                   frame->incoming_id,
                   frame->timestamp,
                   frame->id);
	auto mat = frame->camToWorld.matrix3x4();
	std::cout << mat << "\n";
	odometryFile << mat(0,3) << " " << mat(1,3) << " " << mat(2,3) << std::endl;
        }


        virtual void pushLiveFrame(FrameHessian* image) override
        {
            // can be used to get the raw image / intensity pyramid.
        }

        virtual void pushDepthImage(MinimalImageB3* image) override
        {
            // can be used to get the raw image with depth overlay.
        }
        virtual bool needPushDepthImage() override
        {
            return false;
        }

        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override
        {
            printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
                   KF->frameID,
                   KF->shell->incoming_id,
                   KF->shell->timestamp,
                   KF->shell->id);
            std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

            int maxWrite = 5;
            for(int y=0;y<image->h;y++)
            {
                for(int x=0;x<image->w;x++)
                {
                    if(image->at(x,y) <= 0) continue;

                    printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
                    maxWrite--;
                    if(maxWrite==0) break;
                }
                if(maxWrite==0) break;
            }
        }

    private:
      int numPCL;
      std::ofstream plyFile;
      const std::string plyFileName;
      std::ofstream odometryFile;
      const std::string odometryFileName;
      const bool isSavePCL;
};
}
}
