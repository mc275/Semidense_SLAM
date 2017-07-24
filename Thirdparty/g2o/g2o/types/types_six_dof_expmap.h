// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Modified by Ra��l Mur Artal (2014)
// Added EdgeSE3ProjectXYZ (project using focal_length in x,y directions)
// Modified by Ra��l Mur Artal (2016)
// Added EdgeStereoSE3ProjectXYZ (project using focal_length in x,y directions)
// Added EdgeSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)
// Added EdgeStereoSE3ProjectXYZOnlyPose (unary edge to optimize only the camera pose)

#ifndef G2O_SIX_DOF_TYPES_EXPMAP
#define G2O_SIX_DOF_TYPES_EXPMAP

#include "../core/base_vertex.h"
#include "../core/base_binary_edge.h"
#include "../core/base_unary_edge.h"
#include "se3_ops.h"
#include "se3quat.h"
#include "types_sba.h"
#include <Eigen/Geometry>
#include "../core/g2o_core_api.h"

namespace g2o {
namespace types_six_dof_expmap {
void init();
}

using namespace Eigen;

typedef Matrix<double, 6, 6> Matrix6d;


/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix and externally with its exponential map
 */
class G2O_CORE_API VertexSE3Expmap : public BaseVertex<6, SE3Quat> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3Expmap();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  virtual void setToOriginImpl() {
    _estimate = SE3Quat();
  }

  /**
   * \f$ T \leftarrow exp(\hat{\xi})T \f$
   * @param update_ \f$ \xi \f$ = [w1 w2 w3 v1 v2 v3]
   * @note ���������������
   * @see jlblanco2010geometry3d_techrep.pdf 10.2 \n
   *      ����ҳ�ֻ��ÿ�θ��µõ��������ķ����ǲ�һ����,���ս����ͬ
   */
  virtual void oplusImpl(const double* update_)  {
    Eigen::Map<const Vector6d> update(update_);
    setEstimate(SE3Quat::exp(update)*estimate());
  }
};

/**
 * @brief NOTE uesd in Optimizer::BundleAdjustment(), Optimizer::LocalBundleAdjustment()
 */
class G2O_CORE_API EdgeSE3ProjectXYZ : public  BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  /**
   * ��ͶӰ���
   */
  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]); // T
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]); // Xw
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(v2->estimate()));
  }

  /**
   * ���� \f$ TX_w \f$��Z�Ƿ����0
   * @return true if the depth is Positive
   */
  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }
    
  /**
   * ������[X Y Z]������[w1 w2 w3 v1 v2 v3]���ſ˱Ⱦ��� \n
   * \f$Ji_{2\times3} = \frac{\partial [obs - \Pi(exp(\hat{\xi}) T X_w)]}{\partial X_w} \f$ \n
   * \f$Jj_{2\times6} = \frac{\partial [obs - \Pi(exp(\hat{\xi}) T X_w)]}{\partial \xi} \f$ \n
   * @note _jacobianOplusXi��_jacobianOplusXj
   * @see ������ʽ�������, �ο�(ע�����²ο�����������Ϊ[v1 v2 v3 w1 w2 w3])
   * - jlblanco2010geometry3d_techrep.pdf p56 (A.2) �Ƽ�
   * - strasdat_thesis_2012.pdf p194 (B.4)
   */
  virtual void linearizeOplus();

  /**
   * ��XwͶӰ��cam��ͼ������ϵ \n
   * \f$ u = \frac{f_x X}{Z} + c_x \f$ \n
   * \f$ v = \frac{f_y Y}{Z} + c_v \f$ \n
   * @param  trans_xyz [X Y Z]
   * @return           [u v]
   */
  Vector2d cam_project(const Vector3d & trans_xyz) const;

  double fx, fy, cx, cy;
};

/**
 * @brief NOTE uesd in Optimizer::BundleAdjustment(), Optimizer::LocalBundleAdjustment()
 */
class G2O_CORE_API EdgeStereoSE3ProjectXYZ : public  BaseBinaryEdge<3, Vector3d, VertexSBAPointXYZ, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZ();

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  /**
   * ��ͶӰ���
   */
  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(v2->estimate()),bf);
  }

  /**
   * ���� \f$ TX_w \f$��Z�Ƿ����0
   * @return true if the depth is Positive
   */
  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    return (v1->estimate().map(v2->estimate()))(2)>0.0;
  }

  /**
   * ������[X Y Z]������[w1 w2 w3 v1 v2 v3]���ſ˱Ⱦ��� \n
   * \f$Ji_{3\times3} = \frac{\partial [obs - \Pi(exp(\hat{\xi}) T X_w)]}{\partial X_w} \f$ \n
   * \f$Jj_{3\times6} = \frac{\partial [obs - \Pi(exp(\hat{\xi}) T X_w)]}{\partial \xi} \f$ \n
   * @note _jacobianOplusXi��_jacobianOplusXj
   * @see ������ʽ�������, �ο�(ע�����²ο�����������Ϊ[v1 v2 v3 w1 w2 w3])
   * - jlblanco2010geometry3d_techrep.pdf p56 (A.2) �Ƽ�
   * - strasdat_thesis_2012.pdf p194 (B.4)
   */
  virtual void linearizeOplus();

  /**
   * ��XwͶӰ��cam��ͼ������ϵ \n
   * \f$ ul = \frac{f_x X}{Z} + c_x \f$ \n
   * \f$ vl = \frac{f_y Y}{Z} + c_v \f$ \n
   * \f$ ur = \frac{f_x (X-b)}{Z} + c_x \f$ \n
   * @param  trans_xyz [X Y Z]
   * @param  bf        bf = b*f
   * @return           [ul vl ur]
   */
  Vector3d cam_project(const Vector3d & trans_xyz, const float &bf) const;

  double fx, fy, cx, cy, bf; ///< �ڲ�����bf = b*f
};

// BaseUnaryEdge    �ñ�ֻ��һ������
// 2, Vector2d      ���������Vector2d,���ɶ�Ϊ2
// VertexSE3Expmap  ��������ΪVertexSE3Expmap
// _vertices[0]     ����setVertex(), ��estimateΪTcw,����������ϵ���������ϵ�ı�ʾ
// _measurement     ����setMeasurement, Ϊkeypoint��u,v����
/**
 * @brief NOTE uesd in Optimizer::PoseOptimization()
 */
class G2O_CORE_API EdgeSE3ProjectXYZOnlyPose : public  BaseUnaryEdge<2, Vector2d, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  /**
   * ��ͶӰ���
   */
  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector2d obs(_measurement);
    _error = obs-cam_project(v1->estimate().map(Xw));
  }

  /**
   * ���� \f$ TX_w \f$��Z�Ƿ����0
   * @return true if the depth is Positive
   */
  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }

  /**
   * ����������[w1 w2 w3 v1 v2 v3]���ſ˱Ⱦ���\f$J_{2\times6}\f$
   * \f$ = \frac{\partial [obs - \Pi(exp(\hat{\xi}) T X_w)]}{\partial \xi} \f$
   * @note _jacobianOplusXi���ſɱȾ����������û�й�ϵ
   * @see ������ʽ�������, �ο�(ע�����²ο�����������Ϊ[v1 v2 v3 w1 w2 w3])
   * - jlblanco2010geometry3d_techrep.pdf p56 (A.2) �Ƽ�
   * - strasdat_thesis_2012.pdf p194 (B.4)
   */
  virtual void linearizeOplus();

  /**
   * ��XwͶӰ��cam��ͼ������ϵ \n
   * \f$ u = \frac{f_x X}{Z} + c_x \f$ \n
   * \f$ v = \frac{f_y Y}{Z} + c_v \f$ \n
   * @param  trans_xyz [X Y Z]
   * @return           [u v]
   */
  Vector2d cam_project(const Vector3d & trans_xyz) const;

  Vector3d Xw; ///< MapPoint����������ϵ��λ��
  double fx, fy, cx, cy; ///< �ڲ���
};

// BaseUnaryEdge    �ñ�ֻ��һ������
// 3, Vector3d      ���������Vector3d,���ɶ�Ϊ3
// VertexSE3Expmap  ��������ΪVertexSE3Expmap
// _vertices[0]     ����setVertex(), ��estimateΪTcw,����������ϵ���������ϵ�ı�ʾ
// _measurement     ����setMeasurement, Ϊkeypoint��ul,vl,ur����
/**
 * @brief NOTE uesd in Optimizer::PoseOptimization()
 */
class G2O_CORE_API EdgeStereoSE3ProjectXYZOnlyPose : public  BaseUnaryEdge<3, Vector3d, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeStereoSE3ProjectXYZOnlyPose(){}

  bool read(std::istream& is);

  bool write(std::ostream& os) const;

  /**
   * ��ͶӰ���
   */
  void computeError()  {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    Vector3d obs(_measurement);
    _error = obs - cam_project(v1->estimate().map(Xw));
  }

  /**
   * ���� \f$ TX_w \f$��Z�Ƿ����0
   * @return true if the depth is Positive
   */
  bool isDepthPositive() {
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
    return (v1->estimate().map(Xw))(2)>0.0;
  }

  /**
   * ����������[w1 w2 w3 v1 v2 v3]���ſ˱Ⱦ���\f$J_{3\times6}\f$
   * \f$ = \frac{\partial [obs - \Pi(exp(\hat{\xi}) T X_w)]}{\partial \xi} \f$
   * @note _jacobianOplusXi���ſɱȾ����������û�й�ϵ
   * @see ������ʽ�������, �ο�(ע�����²ο�����������Ϊ[v1 v2 v3 w1 w2 w3])
   * - jlblanco2010geometry3d_techrep.pdf p56 (A.2) �Ƽ�
   * - strasdat_thesis_2012.pdf p194 (B.4)
   */
  virtual void linearizeOplus();

  /**
   * ��XwͶӰ��cam��ͼ������ϵ \n
   * \f$ ul = \frac{f_x X}{Z} + c_x \f$ \n
   * \f$ vl = \frac{f_y Y}{Z} + c_v \f$ \n
   * \f$ ur = \frac{f_x (X-b)}{Z} + c_x \f$ \n
   * @param  trans_xyz [X Y Z]
   * @return           [ul vl ur]
   */
  Vector3d cam_project(const Vector3d & trans_xyz) const; // ��XwͶӰ��cam��ͼ������ϵ

  Vector3d Xw; ///< MapPoint����������ϵ��λ��
  double fx, fy, cx, cy, bf; ///< �ڲ�����bf = b*f
};



} // end namespace

#endif
